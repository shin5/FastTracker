#include "tracker/multi_target_tracker.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <limits>

namespace fasttracker {

MultiTargetTracker::MultiTargetTracker(
    int max_targets,
    const UKFParams& ukf_params,
    const AssociationParams& assoc_params,
    const ProcessNoise& process_noise,
    const MeasurementNoise& meas_noise,
    bool use_imm)
    : max_targets_(max_targets),
      ukf_params_(ukf_params),
      assoc_params_(assoc_params),
      process_noise_(process_noise),
      meas_noise_(meas_noise),
      last_update_time_(0.0),
      first_update_(true),
      use_imm_(use_imm),
      imm_gpu_threshold_(200),  // トラック数200以下はCPU、200超過はGPU
      sensor_x_(0.0f),
      sensor_y_(0.0f),
      sensor_z_(0.0f),
      total_updates_(0),
      total_processing_time_(0.0),
      total_measurements_processed_(0),
      // UKF測定更新用デバイスメモリを事前確保（ホットループ内cudaMalloc/cudaFreeを回避）
      d_meas_(max_targets * MEAS_DIM)
{
    // ========================================
    // GPU利用可能性チェック
    // ========================================
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);

    if (!gpu_available) {
        // GPU利用不可の場合、CPU専用モードに切り替え
        imm_gpu_threshold_ = std::numeric_limits<int>::max();

        // エラー情報をログファイルに出力（コピー可能）
        std::ofstream log_file("/tmp/fasttracker_gpu_error.log");
        log_file << "========================================\n";
        log_file << "FastTracker GPU Availability Check\n";
        log_file << "========================================\n";
        log_file << "CUDA Error: " << cudaGetErrorString(cuda_err) << "\n";
        log_file << "Device Count: " << device_count << "\n";
        log_file << "GPU Status: UNAVAILABLE\n";
        log_file << "Fallback Mode: CPU-ONLY\n";
        log_file << "========================================\n";
        log_file << "Note: If you see 'CUDA driver version is insufficient',\n";
        log_file << "your CUDA driver is older than the runtime (12.6.0).\n";
        log_file << "This is normal in WSL2 environments.\n";
        log_file << "The tracker will run in CPU-only mode automatically.\n";
        log_file << "========================================\n";
        log_file.close();

        std::cerr << "\n";
        std::cerr << "========================================\n";
        std::cerr << "GPU UNAVAILABLE - Running in CPU-only mode\n";
        std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_err) << "\n";
        std::cerr << "Error details saved to: /tmp/fasttracker_gpu_error.log\n";
        std::cerr << "========================================\n";
        std::cerr << "\n";
    } else {
        std::cout << "GPU Available: " << device_count << " device(s) detected\n";
    }

    // UKF初期化
    ukf_ = std::make_unique<UKF>(max_targets, ukf_params, process_noise, meas_noise);

    // IMMフィルタ初期化（ハイブリッド: CPU版とGPU版の両方）
    if (use_imm_) {
        imm_cpu_ = std::make_unique<IMMFilter>(3, max_targets, process_noise);
        if (gpu_available) {
            imm_gpu_ = std::make_unique<IMMFilterGPU>(3, max_targets);
        }
    }

    // トラック管理初期化
    track_manager_ = std::make_unique<TrackManager>(assoc_params);
    track_manager_->setMeasurementNoise(meas_noise);

    // データアソシエーション初期化
    // max_measurements must handle all beams detecting + clutter per frame
    int max_measurements = std::max(max_targets * 5, 200);
    data_association_ = std::make_unique<DataAssociation>(max_targets, max_measurements, assoc_params);
    data_association_->setMeasurementNoise(meas_noise);

    std::cout << "MultiTargetTracker initialized" << std::endl;
    std::cout << "  Max targets: " << max_targets << std::endl;
    if (use_imm_) {
        std::cout << "  IMM filter: enabled (Hybrid CPU/GPU mode)" << std::endl;
        std::cout << "  GPU threshold: " << imm_gpu_threshold_ << " tracks" << std::endl;
    } else {
        std::cout << "  IMM filter: disabled" << std::endl;
    }
}

MultiTargetTracker::~MultiTargetTracker() {
}

void MultiTargetTracker::update(const std::vector<Measurement>& measurements,
                                double current_time) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 初回更新の処理
    if (first_update_) {
        last_update_time_ = current_time;
        first_update_ = false;

        // 初期トラックを作成（SNRフィルタリング）
        for (const auto& meas : measurements) {
            // 低SNRの観測（クラッタ）を除外
            if (meas.snr < assoc_params_.min_snr_for_init) {
                continue;
            }
            track_manager_->initializeTrack(meas);
        }

        total_updates_++;
        return;
    }

    // 時間差分
    double dt = current_time - last_update_time_;
    if (dt <= 0.0) dt = 0.1;  // デフォルト

    // === 予測ステップ ===
    auto predict_start = std::chrono::high_resolution_clock::now();
    predictTracks(dt);
    auto predict_end = std::chrono::high_resolution_clock::now();

    // === データアソシエーション ===
    auto assoc_start = std::chrono::high_resolution_clock::now();
    std::vector<Track> tracks = track_manager_->getAllTracks();
    AssociationResult assoc_result = data_association_->associate(tracks, measurements);
    auto assoc_end = std::chrono::high_resolution_clock::now();

    // === 更新ステップ ===
    auto update_start = std::chrono::high_resolution_clock::now();

    // 割り当て結果に基づいてトラックを分類
    std::vector<int> associated_track_indices;
    std::vector<int> associated_meas_indices;
    std::vector<int> unassociated_track_indices;

    for (size_t i = 0; i < tracks.size(); i++) {
        int meas_idx = assoc_result.track_to_meas[i];
        if (meas_idx >= 0) {
            associated_track_indices.push_back(static_cast<int>(i));
            associated_meas_indices.push_back(meas_idx);
        } else {
            unassociated_track_indices.push_back(static_cast<int>(i));
        }
    }

    // === UKF測定更新（バッチ処理） ===
    if (!associated_track_indices.empty()) {
        int num_assoc = static_cast<int>(associated_track_indices.size());

        // 関連付けられたトラックの状態・共分散・測定値を収集
        std::vector<StateVector> assoc_states(num_assoc);
        std::vector<StateCov> assoc_covs(num_assoc);
        std::vector<float> assoc_meas(num_assoc * MEAS_DIM);

        for (int j = 0; j < num_assoc; j++) {
            int ti = associated_track_indices[j];
            int mi = associated_meas_indices[j];
            assoc_states[j] = tracks[ti].state;
            assoc_covs[j] = tracks[ti].covariance;

            // Measurement構造体をfloat配列に変換
            assoc_meas[j * MEAS_DIM + 0] = measurements[mi].range;
            assoc_meas[j * MEAS_DIM + 1] = measurements[mi].azimuth;
            assoc_meas[j * MEAS_DIM + 2] = measurements[mi].elevation;
            assoc_meas[j * MEAS_DIM + 3] = measurements[mi].doppler;
        }

        // GPU UKF測定更新
        ukf_->copyToDevice(assoc_states, assoc_covs);

        // 測定値をデバイスにコピー（事前確保済みd_meas_を使用）
        // num_assoc <= max_targets_ は上流のtrack_manager_->getAllTracksで保証
        d_meas_.copyFrom(assoc_meas.data(), static_cast<size_t>(num_assoc) * MEAS_DIM);

        // UKF更新ステップ実行（センサー位置を渡す、3D）
        ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                     d_meas_.get(), num_assoc, sensor_x_, sensor_y_, sensor_z_);

        // 結果をホストに戻す
        ukf_->copyToHost(assoc_states, assoc_covs, num_assoc);

        // === IMM モデル確率更新（観測尤度ベース） ===
        if (use_imm_ && imm_cpu_ && static_cast<int>(tracks.size()) < imm_gpu_threshold_) {
            // track_indices: IMM predict時のインデックス = tracks配列内のインデックス
            std::vector<int> imm_track_indices(num_assoc);
            std::vector<Measurement> assoc_measurements(num_assoc);
            for (int j = 0; j < num_assoc; j++) {
                imm_track_indices[j] = associated_track_indices[j];
                assoc_measurements[j] = measurements[associated_meas_indices[j]];
            }

            // トラックのモデル確率を一括取得
            std::vector<float> all_model_probs(tracks.size() * 3);
            for (size_t ti = 0; ti < tracks.size(); ti++) {
                if (tracks[ti].model_probs.size() == 3) {
                    all_model_probs[ti * 3 + 0] = tracks[ti].model_probs[0];
                    all_model_probs[ti * 3 + 1] = tracks[ti].model_probs[1];
                    all_model_probs[ti * 3 + 2] = tracks[ti].model_probs[2];
                } else {
                    all_model_probs[ti * 3 + 0] = 1.0f / 3.0f;
                    all_model_probs[ti * 3 + 1] = 1.0f / 3.0f;
                    all_model_probs[ti * 3 + 2] = 1.0f / 3.0f;
                }
            }

            imm_cpu_->updateModelProbabilities(
                imm_track_indices, assoc_measurements,
                all_model_probs, sensor_x_, sensor_y_, sensor_z_);

            // 更新されたモデル確率をトラックに書き戻し
            for (int j = 0; j < num_assoc; j++) {
                int ti = associated_track_indices[j];
                Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                track.model_probs = {
                    all_model_probs[ti * 3 + 0],
                    all_model_probs[ti * 3 + 1],
                    all_model_probs[ti * 3 + 2]
                };
            }
        }

        // トラックマネージャに反映
        for (int j = 0; j < num_assoc; j++) {
            int ti = associated_track_indices[j];
            if (isStateValid(assoc_states[j], assoc_covs[j])) {
                track_manager_->updateTrack(tracks[ti].id,
                                            assoc_states[j], assoc_covs[j],
                                            current_time);
            } else {
                // UKF更新結果が異常値 → 今フレームの予測状態を維持してミス扱い
                track_manager_->predictOnlyTrack(tracks[ti].id,
                                                 tracks[ti].state,
                                                 tracks[ti].covariance,
                                                 current_time);
            }
        }
    }

    // 観測が割り当てられなかったトラック: 予測のみ
    for (int ti : unassociated_track_indices) {
        track_manager_->predictOnlyTrack(tracks[ti].id,
                                         tracks[ti].state,
                                         tracks[ti].covariance,
                                         current_time);
    }

    // 新規トラック初期化
    initializeNewTracks(assoc_result.unassigned_measurements, measurements);

    // 消失トラック削除
    pruneTracks();

    auto update_end = std::chrono::high_resolution_clock::now();

    // === パフォーマンス統計 ===
    last_perf_stats_.predict_time_ms = std::chrono::duration<double, std::milli>(
        predict_end - predict_start).count();
    last_perf_stats_.association_time_ms = std::chrono::duration<double, std::milli>(
        assoc_end - assoc_start).count();
    last_perf_stats_.update_time_ms = std::chrono::duration<double, std::milli>(
        update_end - update_start).count();
    last_perf_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
        update_end - start_time).count();
    last_perf_stats_.num_tracks = static_cast<int>(tracks.size());
    last_perf_stats_.num_measurements = static_cast<int>(measurements.size());

    // 統計更新
    last_update_time_ = current_time;
    total_updates_++;
    total_processing_time_ += last_perf_stats_.total_time_ms;
    total_measurements_processed_ += static_cast<int>(measurements.size());
}

void MultiTargetTracker::predictTracks(double dt) {
    std::vector<Track> tracks = track_manager_->getAllTracks();
    if (tracks.empty()) return;

    // トラック数がmax_targetsを超える場合は制限
    if (tracks.size() > static_cast<size_t>(max_targets_)) {
        std::cerr << "Warning: Track count (" << tracks.size()
                  << ") exceeds max_targets (" << max_targets_
                  << "). Limiting to max_targets." << std::endl;
        tracks.resize(max_targets_);
    }

    // GPU用に状態と共分散を準備
    std::vector<StateVector> states;
    std::vector<StateCov> covariances;
    std::vector<float> model_probs;

    for (const auto& track : tracks) {
        states.push_back(track.state);
        covariances.push_back(track.covariance);

        // モデル確率を抽出（IMMフィルタ用）
        if (use_imm_ && track.model_probs.size() == 3) {
            for (float prob : track.model_probs) {
                model_probs.push_back(prob);
            }
        } else if (use_imm_) {
            // デフォルト均等確率
            model_probs.push_back(1.0f / 3.0f);
            model_probs.push_back(1.0f / 3.0f);
            model_probs.push_back(1.0f / 3.0f);
        }
    }

    if (use_imm_ && static_cast<int>(tracks.size()) < imm_gpu_threshold_) {
        // ========================================
        // 少数トラック: CPU版IMMで高速処理
        // ========================================
        std::vector<StateVector> predicted_states;
        std::vector<StateCov> predicted_covs;
        std::vector<float> updated_probs;

        imm_cpu_->predict(states, covariances, model_probs,
                         predicted_states, predicted_covs, updated_probs,
                         static_cast<int>(tracks.size()), static_cast<float>(dt));

        // トラックマネージャに反映
        for (size_t i = 0; i < tracks.size(); i++) {
            Track& track = track_manager_->getTrackMutable(tracks[i].id);
            // 異常値（NaN/Inf/物理範囲外）の場合は直前状態を維持
            if (isStateValid(predicted_states[i], predicted_covs[i])) {
                track.state = predicted_states[i];
                track.covariance = predicted_covs[i];
            }

            // モデル確率を更新
            if (updated_probs.size() >= (i + 1) * 3) {
                track.model_probs.clear();
                track.model_probs.push_back(updated_probs[i * 3]);
                track.model_probs.push_back(updated_probs[i * 3 + 1]);
                track.model_probs.push_back(updated_probs[i * 3 + 2]);
            }
        }
    } else if (use_imm_ && static_cast<int>(tracks.size()) >= imm_gpu_threshold_) {
        // ========================================
        // 大量トラック: 標準GPU UKF（IMMなし）で超高速処理
        // GPU IMMはオーバーヘッドが大きいため、大量トラック時は標準UKFが最適
        // ========================================
        ukf_->copyToDevice(states, covariances);
        ukf_->predict(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                      static_cast<int>(tracks.size()), static_cast<float>(dt));
        ukf_->copyToHost(states, covariances, static_cast<int>(tracks.size()));

        for (size_t i = 0; i < tracks.size(); i++) {
            Track& track = track_manager_->getTrackMutable(tracks[i].id);
            // 異常値（NaN/Inf/物理範囲外）の場合は直前状態を維持
            if (isStateValid(states[i], covariances[i])) {
                track.state = states[i];
                track.covariance = covariances[i];
            }
        }
    } else {
        // ========================================
        // IMMなし: 標準GPU UKF
        // ========================================
        ukf_->copyToDevice(states, covariances);
        ukf_->predict(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                      static_cast<int>(tracks.size()), static_cast<float>(dt));
        ukf_->copyToHost(states, covariances, static_cast<int>(tracks.size()));

        for (size_t i = 0; i < tracks.size(); i++) {
            Track& track = track_manager_->getTrackMutable(tracks[i].id);
            // 異常値（NaN/Inf/物理範囲外）の場合は直前状態を維持
            if (isStateValid(states[i], covariances[i])) {
                track.state = states[i];
                track.covariance = covariances[i];
            }
        }
    }
}

void MultiTargetTracker::initializeNewTracks(
    const std::vector<int>& unassigned_measurements,
    const std::vector<Measurement>& measurements)
{
    for (int idx : unassigned_measurements) {
        // トラック数の上限チェック
        if (track_manager_->getNumTracks() >= max_targets_) {
            break;  // これ以上トラックを生成しない
        }

        const auto& meas = measurements[idx];

        // SNRベースのフィルタリング：低品質観測（クラッタ）を除外
        if (meas.snr < assoc_params_.min_snr_for_init) {
            continue;  // クラッタと判断して新規トラック生成をスキップ
        }

        track_manager_->initializeTrack(meas);
    }
}

void MultiTargetTracker::pruneTracks() {
    track_manager_->pruneLostTracks();
}

std::vector<Track> MultiTargetTracker::getConfirmedTracks() const {
    return track_manager_->getConfirmedTracks();
}

std::vector<Track> MultiTargetTracker::getAllTracks() const {
    return track_manager_->getAllTracks();
}

int MultiTargetTracker::getNumTracks() const {
    return track_manager_->getNumTracks();
}

int MultiTargetTracker::getNumConfirmedTracks() const {
    return track_manager_->getNumConfirmedTracks();
}

void MultiTargetTracker::resetStatistics() {
    total_updates_ = 0;
    total_processing_time_ = 0.0;
    total_measurements_processed_ = 0;
    track_manager_->resetStatistics();
}

void MultiTargetTracker::printStatistics() const {
    std::cout << "=== Multi-Target Tracker Statistics ===" << std::endl;
    std::cout << "Total updates: " << total_updates_ << std::endl;
    std::cout << "Total processing time: " << total_processing_time_ << " ms" << std::endl;

    if (total_updates_ > 0) {
        std::cout << "Average processing time: "
                  << (total_processing_time_ / total_updates_) << " ms/frame" << std::endl;
    }

    std::cout << "Total measurements processed: " << total_measurements_processed_ << std::endl;
    std::cout << "Current tracks: " << getNumTracks() << std::endl;
    std::cout << "Confirmed tracks: " << getNumConfirmedTracks() << std::endl;

    std::cout << "\nLast frame performance:" << std::endl;
    std::cout << "  Predict: " << last_perf_stats_.predict_time_ms << " ms" << std::endl;
    std::cout << "  Association: " << last_perf_stats_.association_time_ms << " ms" << std::endl;
    std::cout << "  Update: " << last_perf_stats_.update_time_ms << " ms" << std::endl;
    std::cout << "  Total: " << last_perf_stats_.total_time_ms << " ms" << std::endl;

    std::cout << "=======================================" << std::endl;

    track_manager_->printStatistics();
}

void MultiTargetTracker::setIMMTransitionMatrix(const std::vector<std::vector<float>>& matrix) {
    if (use_imm_) {
        if (imm_cpu_) {
            imm_cpu_->setTransitionMatrix(matrix);
        }
        // GPU version would also need updating if we expose its transition matrix
        // Currently GPU uses hardcoded values in kernels
    }
}

void MultiTargetTracker::setIMMNoiseMultipliers(float cv_mult, float bal_mult, float ct_mult) {
    if (use_imm_) {
        if (imm_cpu_) {
            imm_cpu_->setModelNoiseMultipliers(cv_mult, bal_mult, ct_mult);
        }
        // GPU version would also need updating
    }
}

} // namespace fasttracker
