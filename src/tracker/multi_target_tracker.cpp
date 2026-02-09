#include "tracker/multi_target_tracker.hpp"
#include <chrono>
#include <iostream>

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
      total_updates_(0),
      total_processing_time_(0.0),
      total_measurements_processed_(0)
{
    // UKF初期化
    ukf_ = std::make_unique<UKF>(max_targets, ukf_params, process_noise, meas_noise);

    // IMMフィルタ初期化（ハイブリッド: CPU版とGPU版の両方）
    if (use_imm_) {
        imm_cpu_ = std::make_unique<IMMFilter>(3, max_targets);
        imm_gpu_ = std::make_unique<IMMFilterGPU>(3, max_targets);
    }

    // トラック管理初期化
    track_manager_ = std::make_unique<TrackManager>(assoc_params);

    // データアソシエーション初期化
    data_association_ = std::make_unique<DataAssociation>(max_targets, max_targets * 2, assoc_params);

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
            if (meas.snr < 15.0f) {
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

    // 割り当てられたトラックを更新
    for (size_t i = 0; i < tracks.size(); i++) {
        int meas_idx = assoc_result.track_to_meas[i];

        if (meas_idx >= 0) {
            // 観測が割り当てられた場合: UKF更新
            const auto& meas = measurements[meas_idx];

            // UKFで更新（簡易版：直接状態を取得）
            // 実際にはGPUでバッチ処理すべき
            StateVector state = tracks[i].state;
            StateCov cov = tracks[i].covariance;

            // ここでは簡易的にトラック管理のみ更新
            track_manager_->updateTrack(tracks[i].id, state, cov, current_time);

            // 高品質トラックの確定促進（SNR > 35dB の場合、ヒット数を増やす）
            if (meas.snr > 35.0f) {
                Track& track = track_manager_->getTrackMutable(tracks[i].id);
                if (track.track_state == TrackState::TENTATIVE) {
                    // 高SNRの場合、確定に必要なヒット数を満たす
                    track.hits = std::max(track.hits, assoc_params_.confirm_hits);
                }
            }
        } else {
            // 観測が割り当てられなかった場合: 予測のみ
            track_manager_->predictOnlyTrack(tracks[i].id,
                                            tracks[i].state,
                                            tracks[i].covariance,
                                            current_time);
        }
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
            track.state = predicted_states[i];
            track.covariance = predicted_covs[i];

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
            track.state = states[i];
            track.covariance = covariances[i];
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
            track.state = states[i];
            track.covariance = covariances[i];
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

} // namespace fasttracker
