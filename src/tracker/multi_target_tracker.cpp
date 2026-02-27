#include "tracker/multi_target_tracker.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <limits>
#include <set>

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
        imm_cpu_ = std::make_unique<IMMFilter>(4, max_targets, process_noise);
        if (gpu_available) {
            imm_gpu_ = std::make_unique<IMMFilterGPU>(4, max_targets);
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

    // JPDA初期化（JPDAモード選択時）
    if (assoc_params.association_method == AssociationMethod::JPDA) {
        jpda_association_ = std::make_unique<JPDAAssociation>(assoc_params, meas_noise);
    }

    // PMBM初期化（PMBMモード選択時）
    if (assoc_params.association_method == AssociationMethod::PMBM) {
        pmbm_association_ = std::make_unique<PMBMAssociation>(assoc_params, meas_noise);
    }

    // MHT初期化（MHTモード選択時）
    if (assoc_params.association_method == AssociationMethod::MHT) {
        mht_association_ = std::make_unique<MHTAssociation>(assoc_params, meas_noise);
    }

    // GLMB初期化（GLMBモード選択時）
    if (assoc_params.association_method == AssociationMethod::GLMB) {
        glmb_association_ = std::make_unique<GLMBAssociation>(assoc_params, meas_noise);
    }

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

    // === JPDA / GNN 分岐 ===
    if (assoc_params_.association_method == AssociationMethod::JPDA && jpda_association_) {
        // ========================================
        // JPDA モード: β重み計算 → 合成観測 → GPU UKF更新
        // ========================================
        JPDAResult jpda_result = jpda_association_->associate(
            tracks, measurements, sensor_x_, sensor_y_, sensor_z_);
        auto assoc_end_jpda = std::chrono::high_resolution_clock::now();

        auto update_start_jpda = std::chrono::high_resolution_clock::now();

        // JPDA: best-β実観測でUKF更新（観測共有可: 1-to-1制約なし）
        std::vector<int> jpda_assoc_track_indices;
        std::vector<int> jpda_assoc_meas_indices;
        std::vector<int> jpda_unassoc_track_indices;

        for (auto& upd : jpda_result.track_updates) {
            int ti = upd.track_index;
            if (upd.has_gated_meas && upd.beta_0 < 0.9f && upd.best_meas_index >= 0) {
                jpda_assoc_track_indices.push_back(ti);
                jpda_assoc_meas_indices.push_back(upd.best_meas_index);
            } else {
                jpda_unassoc_track_indices.push_back(ti);
            }
        }

        // === GPU UKF バッチ更新（best-β実観測: 1-to-1制約なし） ===
        if (!jpda_assoc_track_indices.empty()) {
            int num_assoc = static_cast<int>(jpda_assoc_track_indices.size());

            std::vector<StateVector> assoc_states(num_assoc);
            std::vector<StateCov> assoc_covs(num_assoc);
            std::vector<float> assoc_meas(num_assoc * MEAS_DIM);

            for (int j = 0; j < num_assoc; j++) {
                int ti = jpda_assoc_track_indices[j];
                int mi = jpda_assoc_meas_indices[j];
                assoc_states[j] = tracks[ti].state;
                assoc_covs[j] = tracks[ti].covariance;

                // best-β実観測（合成観測はUKF非線形更新と相性が悪い）
                assoc_meas[j * MEAS_DIM + 0] = measurements[mi].range;
                assoc_meas[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                assoc_meas[j * MEAS_DIM + 2] = measurements[mi].elevation;
                assoc_meas[j * MEAS_DIM + 3] = measurements[mi].doppler;
            }

            // GPU UKF測定更新（GNNと同じパス）
            ukf_->copyToDevice(assoc_states, assoc_covs);
            d_meas_.copyFrom(assoc_meas.data(), static_cast<size_t>(num_assoc) * MEAS_DIM);
            ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                         d_meas_.get(), num_assoc, sensor_x_, sensor_y_, sensor_z_);
            ukf_->copyToHost(assoc_states, assoc_covs, num_assoc);

            // === IMM モデル確率更新 ===
            if (use_imm_ && imm_cpu_) {
                std::vector<int> imm_track_indices;
                std::vector<Measurement> imm_measurements;
                for (int j = 0; j < num_assoc; j++) {
                    int mi = jpda_assoc_meas_indices[j];
                    if (mi >= 0) {
                        imm_track_indices.push_back(jpda_assoc_track_indices[j]);
                        imm_measurements.push_back(measurements[mi]);
                    }
                }

                if (!imm_track_indices.empty()) {
                    std::vector<float> all_model_probs(tracks.size() * 4);
                    for (size_t ti = 0; ti < tracks.size(); ti++) {
                        if (tracks[ti].model_probs.size() == 4) {
                            all_model_probs[ti * 4 + 0] = tracks[ti].model_probs[0];
                            all_model_probs[ti * 4 + 1] = tracks[ti].model_probs[1];
                            all_model_probs[ti * 4 + 2] = tracks[ti].model_probs[2];
                            all_model_probs[ti * 4 + 3] = tracks[ti].model_probs[3];
                        } else {
                            all_model_probs[ti * 4 + 0] = 0.25f;
                            all_model_probs[ti * 4 + 1] = 0.25f;
                            all_model_probs[ti * 4 + 2] = 0.25f;
                            all_model_probs[ti * 4 + 3] = 0.25f;
                        }
                    }

                    imm_cpu_->updateModelProbabilities(
                        imm_track_indices, imm_measurements,
                        all_model_probs, sensor_x_, sensor_y_, sensor_z_);

                    for (size_t j = 0; j < imm_track_indices.size(); j++) {
                        int ti = imm_track_indices[j];
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.model_probs = {
                            all_model_probs[ti * 4 + 0],
                            all_model_probs[ti * 4 + 1],
                            all_model_probs[ti * 4 + 2],
                            all_model_probs[ti * 4 + 3]
                        };
                    }
                }
            }

            // UKF更新結果をトラックマネージャに反映
            for (int j = 0; j < num_assoc; j++) {
                int ti = jpda_assoc_track_indices[j];
                if (isStateValid(assoc_states[j], assoc_covs[j])) {
                    // 位置ジャンプチェック
                    float dx = assoc_states[j](0) - tracks[ti].state(0);
                    float dy = assoc_states[j](1) - tracks[ti].state(1);
                    float dz = assoc_states[j](2) - tracks[ti].state(2);
                    float jump_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    float max_jump_vel = assoc_params_.max_jump_velocity;
                    float max_jump = (max_jump_vel > 0.0f)
                        ? std::max(max_jump_vel * static_cast<float>(dt) * 5.0f, 10000.0f)
                        : std::numeric_limits<float>::max();

                    if (jump_dist > max_jump) {
                        track_manager_->predictOnlyTrack(tracks[ti].id,
                                                         tracks[ti].state,
                                                         tracks[ti].covariance,
                                                         current_time);
                    } else {
                        // SNR蓄積
                        int best_mi = jpda_assoc_meas_indices[j];
                        if (best_mi >= 0) {
                            Track& pre_track = track_manager_->getTrackMutable(tracks[ti].id);
                            pre_track.snr_sum += measurements[best_mi].snr;
                        }

                        track_manager_->updateTrack(tracks[ti].id,
                                                    assoc_states[j], assoc_covs[j],
                                                    current_time);
                    }
                } else {
                    track_manager_->predictOnlyTrack(tracks[ti].id,
                                                     tracks[ti].state,
                                                     tracks[ti].covariance,
                                                     current_time);
                }
            }
        }

        // === JPDA CONFIRMED救済: 未割当CONFIRMED航跡を未使用高SNR観測で更新 ===
        std::set<int> jpda_used_meas;
        for (int mi : jpda_assoc_meas_indices) {
            if (mi >= 0) jpda_used_meas.insert(mi);
        }

        std::vector<int> jpda_truly_unassoc;
        for (int ti : jpda_unassoc_track_indices) {
            const Track& t = tracks[ti];
            if (t.track_state != TrackState::CONFIRMED || t.misses < 1) {
                jpda_truly_unassoc.push_back(ti);
                continue;
            }

            // CONFIRMED航跡の予測観測を計算
            float tdx = t.state(0) - sensor_x_;
            float tdy = t.state(1) - sensor_y_;
            float tdz = t.state(2) - sensor_z_;
            float t_range = std::sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
            float t_az = std::atan2(tdy, tdx);
            float t_el_horiz = std::sqrt(tdx*tdx + tdy*tdy);
            float t_el = (t_el_horiz > 1e-6f) ? std::atan2(tdz, t_el_horiz) : 0.0f;

            int best_mi = -1;
            float best_cost = 25.0f;
            float rescue_snr_threshold = assoc_params_.min_snr_for_init + 2.0f;
            for (int mi : jpda_result.base_result.unassigned_measurements) {
                if (jpda_used_meas.count(mi)) continue;
                if (measurements[mi].snr < rescue_snr_threshold) continue;

                float dr = measurements[mi].range - t_range;
                float daz = measurements[mi].azimuth - t_az;
                if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
                if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
                float del = measurements[mi].elevation - t_el;

                float cost = std::sqrt(
                    (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                    (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                    (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise)
                );
                if (cost < best_cost) {
                    best_cost = cost;
                    best_mi = mi;
                }
            }

            if (best_mi >= 0) {
                StateVector rescue_state = t.state;
                StateCov rescue_cov = t.covariance;
                std::vector<StateVector> r_states = {rescue_state};
                std::vector<StateCov> r_covs = {rescue_cov};
                std::vector<float> r_meas = {
                    measurements[best_mi].range,
                    measurements[best_mi].azimuth,
                    measurements[best_mi].elevation,
                    measurements[best_mi].doppler
                };

                ukf_->copyToDevice(r_states, r_covs);
                d_meas_.copyFrom(r_meas.data(), MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), 1, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(r_states, r_covs, 1);

                if (isStateValid(r_states[0], r_covs[0])) {
                    Track& pre_track = track_manager_->getTrackMutable(t.id);
                    pre_track.snr_sum += measurements[best_mi].snr;
                    track_manager_->updateTrack(t.id, r_states[0], r_covs[0], current_time);
                    jpda_used_meas.insert(best_mi);
                } else {
                    jpda_truly_unassoc.push_back(ti);
                }
            } else {
                jpda_truly_unassoc.push_back(ti);
            }
        }

        // 未割当トラック: 予測のみ
        for (int ti : jpda_truly_unassoc) {
            track_manager_->predictOnlyTrack(tracks[ti].id,
                                             tracks[ti].state,
                                             tracks[ti].covariance,
                                             current_time);
        }

        // === JPDA ソフト共有: CONFIRMED クラスタ航跡のドリフト防止 ===
        {
            const float SOFT_SHARE_CLUSTER_SQ = 10000.0f * 10000.0f;
            const float SOFT_SHARE_MEAS_SQ = 5000.0f * 5000.0f;
            const float BLEND_ALPHA = 0.2f;

            std::vector<int> soft_share_tracks;
            std::vector<int> soft_share_meas;

            for (int ti : jpda_truly_unassoc) {
                if (tracks[ti].track_state != TrackState::CONFIRMED) continue;
                if (tracks[ti].misses < 2) continue;

                bool in_cluster = false;
                for (int aj : jpda_assoc_track_indices) {
                    if (tracks[aj].track_state != TrackState::CONFIRMED) continue;
                    float dx = tracks[ti].state(0) - tracks[aj].state(0);
                    float dy = tracks[ti].state(1) - tracks[aj].state(1);
                    float dz = tracks[ti].state(2) - tracks[aj].state(2);
                    if (dx*dx + dy*dy + dz*dz < SOFT_SHARE_CLUSTER_SQ) {
                        in_cluster = true;
                        break;
                    }
                }
                if (!in_cluster) continue;

                int best_meas = -1;
                float best_dist_sq = SOFT_SHARE_MEAS_SQ;
                for (int mi : jpda_assoc_meas_indices) {
                    if (mi < 0) continue;
                    float r = measurements[mi].range;
                    float az = measurements[mi].azimuth;
                    float el = measurements[mi].elevation;
                    float r_h = r * std::cos(el);
                    float mx = r_h * std::cos(az) + sensor_x_;
                    float my = r_h * std::sin(az) + sensor_y_;
                    float mz = r * std::sin(el) + sensor_z_;

                    float dx = mx - tracks[ti].state(0);
                    float dy = my - tracks[ti].state(1);
                    float dz = mz - tracks[ti].state(2);
                    float dist_sq = dx*dx + dy*dy + dz*dz;

                    if (dist_sq < best_dist_sq) {
                        best_dist_sq = dist_sq;
                        best_meas = mi;
                    }
                }

                if (best_meas >= 0) {
                    soft_share_tracks.push_back(ti);
                    soft_share_meas.push_back(best_meas);
                }
            }

            if (!soft_share_tracks.empty()) {
                int num_soft = static_cast<int>(soft_share_tracks.size());
                std::vector<StateVector> soft_states(num_soft);
                std::vector<StateCov> soft_covs(num_soft);
                std::vector<float> soft_meas_data(num_soft * MEAS_DIM);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    int mi = soft_share_meas[j];
                    soft_states[j] = tracks[ti].state;
                    soft_covs[j] = tracks[ti].covariance;
                    soft_meas_data[j * MEAS_DIM + 0] = measurements[mi].range;
                    soft_meas_data[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                    soft_meas_data[j * MEAS_DIM + 2] = measurements[mi].elevation;
                    soft_meas_data[j * MEAS_DIM + 3] = measurements[mi].doppler;
                }

                ukf_->copyToDevice(soft_states, soft_covs);
                d_meas_.copyFrom(soft_meas_data.data(), static_cast<size_t>(num_soft) * MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), num_soft, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(soft_states, soft_covs, num_soft);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    if (isStateValid(soft_states[j], soft_covs[j])) {
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.state = (1.0f - BLEND_ALPHA) * tracks[ti].state
                                    + BLEND_ALPHA * soft_states[j];
                        if (track.hits >= 10 && track.misses > 0) {
                            track.misses--;
                        }
                    }
                }
            }
        }

        // 新規トラック初期化（救済で使用された観測を除外）
        std::vector<int> jpda_final_unassigned_meas;
        for (int mi : jpda_result.base_result.unassigned_measurements) {
            if (!jpda_used_meas.count(mi)) {
                jpda_final_unassigned_meas.push_back(mi);
            }
        }
        initializeNewTracks(jpda_final_unassigned_meas, measurements);

        // CONFIRMED航跡収束プルーニング（無効化: GNNと同様、近接目標の誤削除を防止）
        // pruneConvergedTracks();

        // 消失トラック削除
        pruneTracks();

        auto update_end_jpda = std::chrono::high_resolution_clock::now();

        // パフォーマンス統計
        last_perf_stats_.predict_time_ms = std::chrono::duration<double, std::milli>(
            predict_end - predict_start).count();
        last_perf_stats_.association_time_ms = std::chrono::duration<double, std::milli>(
            assoc_end_jpda - assoc_start).count();
        last_perf_stats_.update_time_ms = std::chrono::duration<double, std::milli>(
            update_end_jpda - update_start_jpda).count();
        last_perf_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            update_end_jpda - start_time).count();
        last_perf_stats_.num_tracks = static_cast<int>(tracks.size());
        last_perf_stats_.num_measurements = static_cast<int>(measurements.size());

        last_update_time_ = current_time;
        total_updates_++;
        total_processing_time_ += last_perf_stats_.total_time_ms;
        total_measurements_processed_ += static_cast<int>(measurements.size());
        return;
    }

    // ========================================
    // PMBM モード
    // ========================================
    if (assoc_params_.association_method == AssociationMethod::PMBM && pmbm_association_) {
        PMBMResult pmbm_result = pmbm_association_->associate(
            tracks, measurements, sensor_x_, sensor_y_, sensor_z_);
        auto assoc_end_pmbm = std::chrono::high_resolution_clock::now();

        auto update_start_pmbm = std::chrono::high_resolution_clock::now();

        // PMBM: ベスト仮説割当でUKF更新（GNNと同じパス）
        std::vector<int> pmbm_assoc_track_indices;
        std::vector<int> pmbm_assoc_meas_indices;
        std::vector<int> pmbm_unassoc_track_indices;

        for (auto& upd : pmbm_result.track_updates) {
            int ti = upd.track_index;
            if (upd.best_meas_index >= 0) {
                pmbm_assoc_track_indices.push_back(ti);
                pmbm_assoc_meas_indices.push_back(upd.best_meas_index);
            } else {
                pmbm_unassoc_track_indices.push_back(ti);
            }
        }

        // === 近接CONFIRMED航跡の割当局所補正 (2-opt swap) ===
        {
            const float CLUSTER_DIST_SQ = 5000.0f * 5000.0f;

            auto computeCostPMBM = [&](int ti, int mi) -> float {
                float dx = tracks[ti].state(0) - sensor_x_;
                float dy = tracks[ti].state(1) - sensor_y_;
                float dz = tracks[ti].state(2) - sensor_z_;
                float pred_range = std::sqrt(dx*dx + dy*dy + dz*dz);
                float pred_az = std::atan2(dy, dx);
                float rh = std::sqrt(dx*dx + dy*dy);
                float pred_el = (rh > 1e-6f) ? std::atan2(dz, rh) : 0.0f;

                float dr = measurements[mi].range - pred_range;
                float daz = measurements[mi].azimuth - pred_az;
                if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
                if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
                float del = measurements[mi].elevation - pred_el;

                return (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                       (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                       (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise);
            };

            bool swapped = true;
            while (swapped) {
                swapped = false;
                for (size_t a = 0; a < pmbm_assoc_track_indices.size(); a++) {
                    for (size_t b = a + 1; b < pmbm_assoc_track_indices.size(); b++) {
                        int ti_a = pmbm_assoc_track_indices[a];
                        int ti_b = pmbm_assoc_track_indices[b];

                        // CONFIRMED航跡のみ2-opt適用（GNNと同条件）
                        if (tracks[ti_a].track_state != TrackState::CONFIRMED) continue;
                        if (tracks[ti_b].track_state != TrackState::CONFIRMED) continue;

                        float ddx = tracks[ti_a].state(0) - tracks[ti_b].state(0);
                        float ddy = tracks[ti_a].state(1) - tracks[ti_b].state(1);
                        float ddz = tracks[ti_a].state(2) - tracks[ti_b].state(2);
                        if (ddx*ddx + ddy*ddy + ddz*ddz >= CLUSTER_DIST_SQ) continue;

                        int mi_a = pmbm_assoc_meas_indices[a];
                        int mi_b = pmbm_assoc_meas_indices[b];

                        float cost_current = computeCostPMBM(ti_a, mi_a) + computeCostPMBM(ti_b, mi_b);
                        float cost_swapped = computeCostPMBM(ti_a, mi_b) + computeCostPMBM(ti_b, mi_a);

                        if (cost_swapped < cost_current * 0.95f) {
                            pmbm_assoc_meas_indices[a] = mi_b;
                            pmbm_assoc_meas_indices[b] = mi_a;
                            swapped = true;
                        }
                    }
                }
            }
        }

        // === GPU UKF バッチ更新 ===
        if (!pmbm_assoc_track_indices.empty()) {
            int num_assoc = static_cast<int>(pmbm_assoc_track_indices.size());

            std::vector<StateVector> assoc_states(num_assoc);
            std::vector<StateCov> assoc_covs(num_assoc);
            std::vector<float> assoc_meas(num_assoc * MEAS_DIM);

            for (int j = 0; j < num_assoc; j++) {
                int ti = pmbm_assoc_track_indices[j];
                int mi = pmbm_assoc_meas_indices[j];
                assoc_states[j] = tracks[ti].state;
                assoc_covs[j] = tracks[ti].covariance;

                assoc_meas[j * MEAS_DIM + 0] = measurements[mi].range;
                assoc_meas[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                assoc_meas[j * MEAS_DIM + 2] = measurements[mi].elevation;
                assoc_meas[j * MEAS_DIM + 3] = measurements[mi].doppler;
            }

            // GPU UKF測定更新
            ukf_->copyToDevice(assoc_states, assoc_covs);
            d_meas_.copyFrom(assoc_meas.data(), static_cast<size_t>(num_assoc) * MEAS_DIM);
            ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                         d_meas_.get(), num_assoc, sensor_x_, sensor_y_, sensor_z_);
            ukf_->copyToHost(assoc_states, assoc_covs, num_assoc);

            // === IMM モデル確率更新 ===
            if (use_imm_ && imm_cpu_) {
                std::vector<int> imm_track_indices;
                std::vector<Measurement> imm_measurements;
                for (int j = 0; j < num_assoc; j++) {
                    int mi = pmbm_assoc_meas_indices[j];
                    if (mi >= 0) {
                        imm_track_indices.push_back(pmbm_assoc_track_indices[j]);
                        imm_measurements.push_back(measurements[mi]);
                    }
                }

                if (!imm_track_indices.empty()) {
                    std::vector<float> all_model_probs(tracks.size() * 4);
                    for (size_t ti = 0; ti < tracks.size(); ti++) {
                        if (tracks[ti].model_probs.size() == 4) {
                            all_model_probs[ti * 4 + 0] = tracks[ti].model_probs[0];
                            all_model_probs[ti * 4 + 1] = tracks[ti].model_probs[1];
                            all_model_probs[ti * 4 + 2] = tracks[ti].model_probs[2];
                            all_model_probs[ti * 4 + 3] = tracks[ti].model_probs[3];
                        } else {
                            all_model_probs[ti * 4 + 0] = 0.25f;
                            all_model_probs[ti * 4 + 1] = 0.25f;
                            all_model_probs[ti * 4 + 2] = 0.25f;
                            all_model_probs[ti * 4 + 3] = 0.25f;
                        }
                    }

                    imm_cpu_->updateModelProbabilities(
                        imm_track_indices, imm_measurements,
                        all_model_probs, sensor_x_, sensor_y_, sensor_z_);

                    for (size_t j = 0; j < imm_track_indices.size(); j++) {
                        int ti = imm_track_indices[j];
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.model_probs = {
                            all_model_probs[ti * 4 + 0],
                            all_model_probs[ti * 4 + 1],
                            all_model_probs[ti * 4 + 2],
                            all_model_probs[ti * 4 + 3]
                        };
                    }
                }
            }

            // UKF更新結果をトラックマネージャに反映
            for (int j = 0; j < num_assoc; j++) {
                int ti = pmbm_assoc_track_indices[j];
                if (isStateValid(assoc_states[j], assoc_covs[j])) {
                    float dx = assoc_states[j](0) - tracks[ti].state(0);
                    float dy = assoc_states[j](1) - tracks[ti].state(1);
                    float dz = assoc_states[j](2) - tracks[ti].state(2);
                    float jump_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    float max_jump_vel = assoc_params_.max_jump_velocity;
                    float max_jump = (max_jump_vel > 0.0f)
                        ? std::max(max_jump_vel * static_cast<float>(dt) * 5.0f, 10000.0f)
                        : std::numeric_limits<float>::max();

                    if (jump_dist > max_jump) {
                        track_manager_->predictOnlyTrack(tracks[ti].id,
                                                         tracks[ti].state,
                                                         tracks[ti].covariance,
                                                         current_time);
                    } else {
                        int best_mi = pmbm_assoc_meas_indices[j];
                        if (best_mi >= 0) {
                            Track& pre_track = track_manager_->getTrackMutable(tracks[ti].id);
                            pre_track.snr_sum += measurements[best_mi].snr;
                        }

                        track_manager_->updateTrack(tracks[ti].id,
                                                    assoc_states[j], assoc_covs[j],
                                                    current_time);
                    }
                } else {
                    track_manager_->predictOnlyTrack(tracks[ti].id,
                                                     tracks[ti].state,
                                                     tracks[ti].covariance,
                                                     current_time);
                }
            }
        }

        // === 存在確率をトラックに書き戻し ===
        for (auto& upd : pmbm_result.track_updates) {
            int ti = upd.track_index;
            if (track_manager_->hasTrack(tracks[ti].id)) {
                Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                track.existence_prob = upd.existence_prob;
            }
        }

        // === PMBM 救済: 未割当CONFIRMED航跡を未使用高SNR観測で更新 ===
        // CONFIRMED航跡のみ救済対象（TENTATIVEは救済しない → 偽トラック確認を抑制）
        // ゲートとSNR閾値を引き締め: クラッタによる不要な救済を防止
        std::set<int> pmbm_used_meas;
        for (int mi : pmbm_assoc_meas_indices) {
            if (mi >= 0) pmbm_used_meas.insert(mi);
        }

        std::vector<int> pmbm_truly_unassoc;
        for (int ti : pmbm_unassoc_track_indices) {
            const Track& t = tracks[ti];
            bool rescue_eligible = (t.track_state == TrackState::CONFIRMED && t.misses >= 1);
            if (!rescue_eligible) {
                pmbm_truly_unassoc.push_back(ti);
                continue;
            }

            float tdx = t.state(0) - sensor_x_;
            float tdy = t.state(1) - sensor_y_;
            float tdz = t.state(2) - sensor_z_;
            float t_range = std::sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
            float t_az = std::atan2(tdy, tdx);
            float t_el_horiz = std::sqrt(tdx*tdx + tdy*tdy);
            float t_el = (t_el_horiz > 1e-6f) ? std::atan2(tdz, t_el_horiz) : 0.0f;

            int best_mi = -1;
            float best_cost = 25.0f;
            float rescue_snr_threshold = assoc_params_.min_snr_for_init + 2.0f;
            for (int mi : pmbm_result.base_result.unassigned_measurements) {
                if (pmbm_used_meas.count(mi)) continue;
                if (measurements[mi].snr < rescue_snr_threshold) continue;

                float dr = measurements[mi].range - t_range;
                float daz = measurements[mi].azimuth - t_az;
                if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
                if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
                float del = measurements[mi].elevation - t_el;

                float cost = std::sqrt(
                    (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                    (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                    (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise)
                );
                if (cost < best_cost) {
                    best_cost = cost;
                    best_mi = mi;
                }
            }

            if (best_mi >= 0) {
                StateVector rescue_state = t.state;
                StateCov rescue_cov = t.covariance;
                std::vector<StateVector> r_states = {rescue_state};
                std::vector<StateCov> r_covs = {rescue_cov};
                std::vector<float> r_meas = {
                    measurements[best_mi].range,
                    measurements[best_mi].azimuth,
                    measurements[best_mi].elevation,
                    measurements[best_mi].doppler
                };

                ukf_->copyToDevice(r_states, r_covs);
                d_meas_.copyFrom(r_meas.data(), MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), 1, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(r_states, r_covs, 1);

                if (isStateValid(r_states[0], r_covs[0])) {
                    Track& pre_track = track_manager_->getTrackMutable(t.id);
                    pre_track.snr_sum += measurements[best_mi].snr;
                    track_manager_->updateTrack(t.id, r_states[0], r_covs[0], current_time);
                    pmbm_used_meas.insert(best_mi);
                } else {
                    pmbm_truly_unassoc.push_back(ti);
                }
            } else {
                pmbm_truly_unassoc.push_back(ti);
            }
        }

        // 未割当トラック: 予測のみ
        for (int ti : pmbm_truly_unassoc) {
            track_manager_->predictOnlyTrack(tracks[ti].id,
                                             tracks[ti].state,
                                             tracks[ti].covariance,
                                             current_time);
        }

        // === PMBM ソフト共有: CONFIRMED クラスタ航跡のドリフト防止 ===
        // 存在確率が高い航跡のみ対象（低存在確率の偽トラックを延命させない）
        {
            const float SOFT_SHARE_CLUSTER_SQ = 10000.0f * 10000.0f;
            const float SOFT_SHARE_MEAS_SQ = 5000.0f * 5000.0f;
            const float BLEND_ALPHA = 0.2f;

            std::vector<int> soft_share_tracks;
            std::vector<int> soft_share_meas;

            for (int ti : pmbm_truly_unassoc) {
                if (tracks[ti].track_state != TrackState::CONFIRMED) continue;
                if (tracks[ti].misses < 2) continue;
                // 存在確率フィルタ: 低存在確率の航跡はソフト共有で延命しない
                if (tracks[ti].existence_prob < 0.3f) continue;

                bool in_cluster = false;
                for (size_t aj = 0; aj < pmbm_assoc_track_indices.size(); aj++) {
                    int ati = pmbm_assoc_track_indices[aj];
                    if (tracks[ati].track_state != TrackState::CONFIRMED) continue;
                    float dx = tracks[ti].state(0) - tracks[ati].state(0);
                    float dy = tracks[ti].state(1) - tracks[ati].state(1);
                    float dz = tracks[ti].state(2) - tracks[ati].state(2);
                    if (dx*dx + dy*dy + dz*dz < SOFT_SHARE_CLUSTER_SQ) {
                        in_cluster = true;
                        break;
                    }
                }
                if (!in_cluster) continue;

                int best_meas = -1;
                float best_dist_sq = SOFT_SHARE_MEAS_SQ;
                for (int mi : pmbm_assoc_meas_indices) {
                    if (mi < 0) continue;
                    float r = measurements[mi].range;
                    float az = measurements[mi].azimuth;
                    float el = measurements[mi].elevation;
                    float r_h = r * std::cos(el);
                    float mx = r_h * std::cos(az) + sensor_x_;
                    float my = r_h * std::sin(az) + sensor_y_;
                    float mz = r * std::sin(el) + sensor_z_;

                    float dx = mx - tracks[ti].state(0);
                    float dy = my - tracks[ti].state(1);
                    float dz = mz - tracks[ti].state(2);
                    float dist_sq = dx*dx + dy*dy + dz*dz;

                    if (dist_sq < best_dist_sq) {
                        best_dist_sq = dist_sq;
                        best_meas = mi;
                    }
                }

                if (best_meas >= 0) {
                    soft_share_tracks.push_back(ti);
                    soft_share_meas.push_back(best_meas);
                }
            }

            if (!soft_share_tracks.empty()) {
                int num_soft = static_cast<int>(soft_share_tracks.size());
                std::vector<StateVector> soft_states(num_soft);
                std::vector<StateCov> soft_covs(num_soft);
                std::vector<float> soft_meas_data(num_soft * MEAS_DIM);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    int mi = soft_share_meas[j];
                    soft_states[j] = tracks[ti].state;
                    soft_covs[j] = tracks[ti].covariance;
                    soft_meas_data[j * MEAS_DIM + 0] = measurements[mi].range;
                    soft_meas_data[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                    soft_meas_data[j * MEAS_DIM + 2] = measurements[mi].elevation;
                    soft_meas_data[j * MEAS_DIM + 3] = measurements[mi].doppler;
                }

                ukf_->copyToDevice(soft_states, soft_covs);
                d_meas_.copyFrom(soft_meas_data.data(), static_cast<size_t>(num_soft) * MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), num_soft, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(soft_states, soft_covs, num_soft);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    if (isStateValid(soft_states[j], soft_covs[j])) {
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.state = (1.0f - BLEND_ALPHA) * tracks[ti].state
                                    + BLEND_ALPHA * soft_states[j];
                        if (track.hits >= 10 && track.misses > 0) {
                            track.misses--;
                        }
                    }
                }
            }
        }

        // 新規トラック初期化（PMBM: CONFIRMED近接フィルタ）
        // CONFIRMED航跡から5km以内の観測を除外（重複トラック抑制）
        std::vector<int> pmbm_final_unassigned_meas;
        {
            const float PMBM_INIT_SUPPRESS_SQ = 5000.0f * 5000.0f;  // 5km
            auto confirmed_for_init = track_manager_->getConfirmedTracks();

            for (int mi : pmbm_result.base_result.unassigned_measurements) {
                if (pmbm_used_meas.count(mi)) continue;

                bool near_confirmed = false;
                if (!confirmed_for_init.empty()) {
                    float r = measurements[mi].range;
                    float az = measurements[mi].azimuth;
                    float el = measurements[mi].elevation;
                    float r_h = r * std::cos(el);
                    float mx = r_h * std::cos(az) + sensor_x_;
                    float my = r_h * std::sin(az) + sensor_y_;
                    float mz = r * std::sin(el) + sensor_z_;

                    for (const auto& ct : confirmed_for_init) {
                        if (ct.misses >= 3) continue;
                        float dx = mx - ct.state(0);
                        float dy = my - ct.state(1);
                        float dz = mz - ct.state(2);
                        if (dx*dx + dy*dy + dz*dz < PMBM_INIT_SUPPRESS_SQ) {
                            near_confirmed = true;
                            break;
                        }
                    }
                }
                if (!near_confirmed) {
                    pmbm_final_unassigned_meas.push_back(mi);
                }
            }
        }
        initializeNewTracks(pmbm_final_unassigned_meas, measurements);

        // 新規トラックに初期存在確率を設定
        {
            auto all_tracks_now = track_manager_->getAllTracks();
            for (const auto& t : all_tracks_now) {
                if (t.age == 0 && t.track_state == TrackState::TENTATIVE) {
                    Track& mt = track_manager_->getTrackMutable(t.id);
                    mt.existence_prob = assoc_params_.pmbm_initial_existence;
                }
            }
        }

        // 収束プルーニング無効化
        // pruneConvergedTracks();

        // 消失トラック削除
        pruneTracks();

        auto update_end_pmbm = std::chrono::high_resolution_clock::now();

        // パフォーマンス統計
        last_perf_stats_.predict_time_ms = std::chrono::duration<double, std::milli>(
            predict_end - predict_start).count();
        last_perf_stats_.association_time_ms = std::chrono::duration<double, std::milli>(
            assoc_end_pmbm - assoc_start).count();
        last_perf_stats_.update_time_ms = std::chrono::duration<double, std::milli>(
            update_end_pmbm - update_start_pmbm).count();
        last_perf_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            update_end_pmbm - start_time).count();
        last_perf_stats_.num_tracks = static_cast<int>(tracks.size());
        last_perf_stats_.num_measurements = static_cast<int>(measurements.size());

        last_update_time_ = current_time;
        total_updates_++;
        total_processing_time_ += last_perf_stats_.total_time_ms;
        total_measurements_processed_ += static_cast<int>(measurements.size());
        return;
    }

    // ========================================
    // MHT モード
    // ========================================
    if (assoc_params_.association_method == AssociationMethod::MHT && mht_association_) {
        MHTResult mht_result = mht_association_->associate(
            tracks, measurements, sensor_x_, sensor_y_, sensor_z_);
        auto assoc_end_mht = std::chrono::high_resolution_clock::now();

        auto update_start_mht = std::chrono::high_resolution_clock::now();

        // MHT: ベスト仮説割当でUKF更新（PMBM/GNNと同じパス）
        std::vector<int> mht_assoc_track_indices;
        std::vector<int> mht_assoc_meas_indices;
        std::vector<int> mht_unassoc_track_indices;

        for (auto& info : mht_result.track_info) {
            int ti = info.track_index;
            if (info.best_meas_index >= 0) {
                mht_assoc_track_indices.push_back(ti);
                mht_assoc_meas_indices.push_back(info.best_meas_index);
            } else {
                mht_unassoc_track_indices.push_back(ti);
            }
        }

        // === 近接CONFIRMED航跡の割当局所補正 (2-opt swap) ===
        {
            const float CLUSTER_DIST_SQ = 5000.0f * 5000.0f;

            auto computeCostMHT = [&](int ti, int mi) -> float {
                float dx = tracks[ti].state(0) - sensor_x_;
                float dy = tracks[ti].state(1) - sensor_y_;
                float dz = tracks[ti].state(2) - sensor_z_;
                float pred_range = std::sqrt(dx*dx + dy*dy + dz*dz);
                float pred_az = std::atan2(dy, dx);
                float rh = std::sqrt(dx*dx + dy*dy);
                float pred_el = (rh > 1e-6f) ? std::atan2(dz, rh) : 0.0f;

                float dr = measurements[mi].range - pred_range;
                float daz = measurements[mi].azimuth - pred_az;
                if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
                if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
                float del = measurements[mi].elevation - pred_el;

                return (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                       (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                       (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise);
            };

            bool swapped = true;
            while (swapped) {
                swapped = false;
                for (size_t a = 0; a < mht_assoc_track_indices.size(); a++) {
                    for (size_t b = a + 1; b < mht_assoc_track_indices.size(); b++) {
                        int ti_a = mht_assoc_track_indices[a];
                        int ti_b = mht_assoc_track_indices[b];

                        if (tracks[ti_a].track_state != TrackState::CONFIRMED) continue;
                        if (tracks[ti_b].track_state != TrackState::CONFIRMED) continue;

                        float ddx = tracks[ti_a].state(0) - tracks[ti_b].state(0);
                        float ddy = tracks[ti_a].state(1) - tracks[ti_b].state(1);
                        float ddz = tracks[ti_a].state(2) - tracks[ti_b].state(2);
                        if (ddx*ddx + ddy*ddy + ddz*ddz >= CLUSTER_DIST_SQ) continue;

                        int mi_a = mht_assoc_meas_indices[a];
                        int mi_b = mht_assoc_meas_indices[b];

                        float cost_current = computeCostMHT(ti_a, mi_a) + computeCostMHT(ti_b, mi_b);
                        float cost_swapped = computeCostMHT(ti_a, mi_b) + computeCostMHT(ti_b, mi_a);

                        if (cost_swapped < cost_current * 0.95f) {
                            mht_assoc_meas_indices[a] = mi_b;
                            mht_assoc_meas_indices[b] = mi_a;
                            swapped = true;
                        }
                    }
                }
            }
        }

        // === GPU UKF バッチ更新 ===
        if (!mht_assoc_track_indices.empty()) {
            int num_assoc = static_cast<int>(mht_assoc_track_indices.size());

            std::vector<StateVector> assoc_states(num_assoc);
            std::vector<StateCov> assoc_covs(num_assoc);
            std::vector<float> assoc_meas(num_assoc * MEAS_DIM);

            for (int j = 0; j < num_assoc; j++) {
                int ti = mht_assoc_track_indices[j];
                int mi = mht_assoc_meas_indices[j];
                assoc_states[j] = tracks[ti].state;
                assoc_covs[j] = tracks[ti].covariance;

                assoc_meas[j * MEAS_DIM + 0] = measurements[mi].range;
                assoc_meas[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                assoc_meas[j * MEAS_DIM + 2] = measurements[mi].elevation;
                assoc_meas[j * MEAS_DIM + 3] = measurements[mi].doppler;
            }

            ukf_->copyToDevice(assoc_states, assoc_covs);
            d_meas_.copyFrom(assoc_meas.data(), static_cast<size_t>(num_assoc) * MEAS_DIM);
            ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                         d_meas_.get(), num_assoc, sensor_x_, sensor_y_, sensor_z_);
            ukf_->copyToHost(assoc_states, assoc_covs, num_assoc);

            // === IMM モデル確率更新 ===
            if (use_imm_ && imm_cpu_) {
                std::vector<int> imm_track_indices;
                std::vector<Measurement> imm_measurements;
                for (int j = 0; j < num_assoc; j++) {
                    int mi = mht_assoc_meas_indices[j];
                    if (mi >= 0) {
                        imm_track_indices.push_back(mht_assoc_track_indices[j]);
                        imm_measurements.push_back(measurements[mi]);
                    }
                }

                if (!imm_track_indices.empty()) {
                    std::vector<float> all_model_probs(tracks.size() * 4);
                    for (size_t ti = 0; ti < tracks.size(); ti++) {
                        if (tracks[ti].model_probs.size() == 4) {
                            all_model_probs[ti * 4 + 0] = tracks[ti].model_probs[0];
                            all_model_probs[ti * 4 + 1] = tracks[ti].model_probs[1];
                            all_model_probs[ti * 4 + 2] = tracks[ti].model_probs[2];
                            all_model_probs[ti * 4 + 3] = tracks[ti].model_probs[3];
                        } else {
                            all_model_probs[ti * 4 + 0] = 0.25f;
                            all_model_probs[ti * 4 + 1] = 0.25f;
                            all_model_probs[ti * 4 + 2] = 0.25f;
                            all_model_probs[ti * 4 + 3] = 0.25f;
                        }
                    }

                    imm_cpu_->updateModelProbabilities(
                        imm_track_indices, imm_measurements,
                        all_model_probs, sensor_x_, sensor_y_, sensor_z_);

                    for (size_t j = 0; j < imm_track_indices.size(); j++) {
                        int ti = imm_track_indices[j];
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.model_probs = {
                            all_model_probs[ti * 4 + 0],
                            all_model_probs[ti * 4 + 1],
                            all_model_probs[ti * 4 + 2],
                            all_model_probs[ti * 4 + 3]
                        };
                    }
                }
            }

            // UKF更新結果をトラックマネージャに反映
            for (int j = 0; j < num_assoc; j++) {
                int ti = mht_assoc_track_indices[j];
                if (isStateValid(assoc_states[j], assoc_covs[j])) {
                    float dx = assoc_states[j](0) - tracks[ti].state(0);
                    float dy = assoc_states[j](1) - tracks[ti].state(1);
                    float dz = assoc_states[j](2) - tracks[ti].state(2);
                    float jump_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    float max_jump_vel = assoc_params_.max_jump_velocity;
                    float max_jump = (max_jump_vel > 0.0f)
                        ? std::max(max_jump_vel * static_cast<float>(dt) * 5.0f, 10000.0f)
                        : std::numeric_limits<float>::max();

                    if (jump_dist > max_jump) {
                        track_manager_->predictOnlyTrack(tracks[ti].id,
                                                         tracks[ti].state,
                                                         tracks[ti].covariance,
                                                         current_time);
                    } else {
                        int best_mi = mht_assoc_meas_indices[j];
                        if (best_mi >= 0) {
                            Track& pre_track = track_manager_->getTrackMutable(tracks[ti].id);
                            pre_track.snr_sum += measurements[best_mi].snr;
                        }

                        track_manager_->updateTrack(tracks[ti].id,
                                                    assoc_states[j], assoc_covs[j],
                                                    current_time);
                    }
                } else {
                    track_manager_->predictOnlyTrack(tracks[ti].id,
                                                     tracks[ti].state,
                                                     tracks[ti].covariance,
                                                     current_time);
                }
            }
        }

        // === MHT: 存在確率write-back（PMBMと同一のベイジアン更新） ===
        for (const auto& info : mht_result.track_info) {
            int ti = info.track_index;
            if (track_manager_->hasTrack(tracks[ti].id)) {
                Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                track.existence_prob = info.existence_prob;
            }
        }

        // === MHT 救済: 未割当航跡を未使用高SNR観測で更新 ===
        // CONFIRMED: miss>=1で救済（PMBMと同一）
        // TENTATIVE: 高SNR航跡も救済（GNNと同一 — 確認高速化）
        std::set<int> mht_used_meas;
        for (int mi : mht_assoc_meas_indices) {
            if (mi >= 0) mht_used_meas.insert(mi);
        }

        std::vector<int> mht_truly_unassoc;
        for (int ti : mht_unassoc_track_indices) {
            const Track& t = tracks[ti];
            bool rescue_eligible = false;
            if (t.track_state == TrackState::CONFIRMED && t.misses >= 1) {
                rescue_eligible = true;
            } else if (t.track_state == TrackState::TENTATIVE &&
                       t.snr_sum / std::max(t.hits, 1) >= assoc_params_.min_snr_for_init + 5.0f) {
                rescue_eligible = true;
            }
            if (!rescue_eligible) {
                mht_truly_unassoc.push_back(ti);
                continue;
            }

            float tdx = t.state(0) - sensor_x_;
            float tdy = t.state(1) - sensor_y_;
            float tdz = t.state(2) - sensor_z_;
            float t_range = std::sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
            float t_az = std::atan2(tdy, tdx);
            float t_el_horiz = std::sqrt(tdx*tdx + tdy*tdy);
            float t_el = (t_el_horiz > 1e-6f) ? std::atan2(tdz, t_el_horiz) : 0.0f;

            int best_mi = -1;
            float best_cost = 25.0f;
            float rescue_snr_threshold = assoc_params_.min_snr_for_init + 2.0f;
            for (int mi : mht_result.base_result.unassigned_measurements) {
                if (mht_used_meas.count(mi)) continue;
                if (measurements[mi].snr < rescue_snr_threshold) continue;

                float dr = measurements[mi].range - t_range;
                float daz = measurements[mi].azimuth - t_az;
                if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
                if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
                float del = measurements[mi].elevation - t_el;

                float cost = std::sqrt(
                    (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                    (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                    (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise)
                );
                if (cost < best_cost) {
                    best_cost = cost;
                    best_mi = mi;
                }
            }

            if (best_mi >= 0) {
                StateVector rescue_state = t.state;
                StateCov rescue_cov = t.covariance;
                std::vector<StateVector> r_states = {rescue_state};
                std::vector<StateCov> r_covs = {rescue_cov};
                std::vector<float> r_meas = {
                    measurements[best_mi].range,
                    measurements[best_mi].azimuth,
                    measurements[best_mi].elevation,
                    measurements[best_mi].doppler
                };

                ukf_->copyToDevice(r_states, r_covs);
                d_meas_.copyFrom(r_meas.data(), MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), 1, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(r_states, r_covs, 1);

                if (isStateValid(r_states[0], r_covs[0])) {
                    Track& pre_track = track_manager_->getTrackMutable(t.id);
                    pre_track.snr_sum += measurements[best_mi].snr;
                    track_manager_->updateTrack(t.id, r_states[0], r_covs[0], current_time);
                    mht_used_meas.insert(best_mi);
                } else {
                    mht_truly_unassoc.push_back(ti);
                }
            } else {
                mht_truly_unassoc.push_back(ti);
            }
        }

        // 未割当トラック: 予測のみ（観測共有なし — PMBMと同等のパス）
        for (int ti : mht_truly_unassoc) {
            track_manager_->predictOnlyTrack(tracks[ti].id,
                                             tracks[ti].state,
                                             tracks[ti].covariance,
                                             current_time);
        }

        // === MHT ソフト共有: CONFIRMED クラスタ航跡のドリフト防止 ===
        {
            const float SOFT_SHARE_CLUSTER_SQ = 10000.0f * 10000.0f;
            const float SOFT_SHARE_MEAS_SQ = 5000.0f * 5000.0f;
            const float BLEND_ALPHA = 0.2f;

            std::vector<int> soft_share_tracks;
            std::vector<int> soft_share_meas;

            for (int ti : mht_truly_unassoc) {
                if (tracks[ti].track_state != TrackState::CONFIRMED) continue;
                if (tracks[ti].misses < 2) continue;
                if (tracks[ti].existence_prob < 0.3f) continue;

                bool in_cluster = false;
                for (size_t aj = 0; aj < mht_assoc_track_indices.size(); aj++) {
                    int ati = mht_assoc_track_indices[aj];
                    if (tracks[ati].track_state != TrackState::CONFIRMED) continue;
                    float dx = tracks[ti].state(0) - tracks[ati].state(0);
                    float dy = tracks[ti].state(1) - tracks[ati].state(1);
                    float dz = tracks[ti].state(2) - tracks[ati].state(2);
                    if (dx*dx + dy*dy + dz*dz < SOFT_SHARE_CLUSTER_SQ) {
                        in_cluster = true;
                        break;
                    }
                }
                if (!in_cluster) continue;

                int best_meas = -1;
                float best_dist_sq = SOFT_SHARE_MEAS_SQ;
                for (int mi : mht_assoc_meas_indices) {
                    if (mi < 0) continue;
                    float r = measurements[mi].range;
                    float az = measurements[mi].azimuth;
                    float el = measurements[mi].elevation;
                    float r_h = r * std::cos(el);
                    float mx = r_h * std::cos(az) + sensor_x_;
                    float my = r_h * std::sin(az) + sensor_y_;
                    float mz = r * std::sin(el) + sensor_z_;

                    float dx = mx - tracks[ti].state(0);
                    float dy = my - tracks[ti].state(1);
                    float dz = mz - tracks[ti].state(2);
                    float dist_sq = dx*dx + dy*dy + dz*dz;

                    if (dist_sq < best_dist_sq) {
                        best_dist_sq = dist_sq;
                        best_meas = mi;
                    }
                }

                if (best_meas >= 0) {
                    soft_share_tracks.push_back(ti);
                    soft_share_meas.push_back(best_meas);
                }
            }

            if (!soft_share_tracks.empty()) {
                int num_soft = static_cast<int>(soft_share_tracks.size());
                std::vector<StateVector> soft_states(num_soft);
                std::vector<StateCov> soft_covs(num_soft);
                std::vector<float> soft_meas_data(num_soft * MEAS_DIM);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    int mi = soft_share_meas[j];
                    soft_states[j] = tracks[ti].state;
                    soft_covs[j] = tracks[ti].covariance;
                    soft_meas_data[j * MEAS_DIM + 0] = measurements[mi].range;
                    soft_meas_data[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                    soft_meas_data[j * MEAS_DIM + 2] = measurements[mi].elevation;
                    soft_meas_data[j * MEAS_DIM + 3] = measurements[mi].doppler;
                }

                ukf_->copyToDevice(soft_states, soft_covs);
                d_meas_.copyFrom(soft_meas_data.data(), static_cast<size_t>(num_soft) * MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), num_soft, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(soft_states, soft_covs, num_soft);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    if (isStateValid(soft_states[j], soft_covs[j])) {
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.state = (1.0f - BLEND_ALPHA) * tracks[ti].state
                                    + BLEND_ALPHA * soft_states[j];
                        if (track.hits >= 10 && track.misses > 0) {
                            track.misses--;
                        }
                    }
                }
            }
        }

        // 新規トラック初期化（MHT: init抑制なし — GNNと同等のオープンな初期化）
        // MHTはGNN等価の割当品質のため、PMBMの5km抑制は不適合。
        // 抑制なしで自由に新規トラックを生成し、pruneConvergedTracksで収束航跡を統合する。
        {
            std::vector<int> mht_final_unassigned_meas;
            for (int mi : mht_result.base_result.unassigned_measurements) {
                if (mht_used_meas.count(mi)) continue;
                mht_final_unassigned_meas.push_back(mi);
            }
            initializeNewTracks(mht_final_unassigned_meas, measurements);
        }

        // 新規トラックに初期存在確率を設定（MHTでもPMBMと同じ初期値を使用）
        {
            auto all_tracks_now = track_manager_->getAllTracks();
            for (const auto& t : all_tracks_now) {
                if (t.age == 0 && t.track_state == TrackState::TENTATIVE) {
                    Track& mt = track_manager_->getTrackMutable(t.id);
                    mt.existence_prob = assoc_params_.pmbm_initial_existence;
                }
            }
        }

        pruneTracks();

        auto update_end_mht = std::chrono::high_resolution_clock::now();

        last_perf_stats_.predict_time_ms = std::chrono::duration<double, std::milli>(
            predict_end - predict_start).count();
        last_perf_stats_.association_time_ms = std::chrono::duration<double, std::milli>(
            assoc_end_mht - assoc_start).count();
        last_perf_stats_.update_time_ms = std::chrono::duration<double, std::milli>(
            update_end_mht - update_start_mht).count();
        last_perf_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            update_end_mht - start_time).count();
        last_perf_stats_.num_tracks = static_cast<int>(tracks.size());
        last_perf_stats_.num_measurements = static_cast<int>(measurements.size());

        last_update_time_ = current_time;
        total_updates_++;
        total_processing_time_ += last_perf_stats_.total_time_ms;
        total_measurements_processed_ += static_cast<int>(measurements.size());
        return;
    }

    // ========================================
    // GLMB モード
    // ========================================
    if (assoc_params_.association_method == AssociationMethod::GLMB && glmb_association_) {
        GLMBResult glmb_result = glmb_association_->associate(
            tracks, measurements, sensor_x_, sensor_y_, sensor_z_);
        auto assoc_end_glmb = std::chrono::high_resolution_clock::now();

        auto update_start_glmb = std::chrono::high_resolution_clock::now();

        // GLMB: ベスト仮説割当でUKF更新
        std::vector<int> glmb_assoc_track_indices;
        std::vector<int> glmb_assoc_meas_indices;
        std::vector<int> glmb_unassoc_track_indices;

        for (auto& upd : glmb_result.track_updates) {
            int ti = upd.track_index;
            if (upd.best_meas_index >= 0) {
                glmb_assoc_track_indices.push_back(ti);
                glmb_assoc_meas_indices.push_back(upd.best_meas_index);
            } else {
                glmb_unassoc_track_indices.push_back(ti);
            }
        }

        // === 近接CONFIRMED航跡の割当局所補正 (2-opt swap) ===
        {
            const float CLUSTER_DIST_SQ = 5000.0f * 5000.0f;

            auto computeCostGLMB = [&](int ti, int mi) -> float {
                float dx = tracks[ti].state(0) - sensor_x_;
                float dy = tracks[ti].state(1) - sensor_y_;
                float dz = tracks[ti].state(2) - sensor_z_;
                float pred_range = std::sqrt(dx*dx + dy*dy + dz*dz);
                float pred_az = std::atan2(dy, dx);
                float rh = std::sqrt(dx*dx + dy*dy);
                float pred_el = (rh > 1e-6f) ? std::atan2(dz, rh) : 0.0f;

                float dr = measurements[mi].range - pred_range;
                float daz = measurements[mi].azimuth - pred_az;
                if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
                if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
                float del = measurements[mi].elevation - pred_el;

                return (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                       (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                       (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise);
            };

            bool swapped = true;
            while (swapped) {
                swapped = false;
                for (size_t a = 0; a < glmb_assoc_track_indices.size(); a++) {
                    for (size_t b = a + 1; b < glmb_assoc_track_indices.size(); b++) {
                        int ti_a = glmb_assoc_track_indices[a];
                        int ti_b = glmb_assoc_track_indices[b];

                        if (tracks[ti_a].track_state != TrackState::CONFIRMED) continue;
                        if (tracks[ti_b].track_state != TrackState::CONFIRMED) continue;

                        float ddx = tracks[ti_a].state(0) - tracks[ti_b].state(0);
                        float ddy = tracks[ti_a].state(1) - tracks[ti_b].state(1);
                        float ddz = tracks[ti_a].state(2) - tracks[ti_b].state(2);
                        if (ddx*ddx + ddy*ddy + ddz*ddz >= CLUSTER_DIST_SQ) continue;

                        int mi_a = glmb_assoc_meas_indices[a];
                        int mi_b = glmb_assoc_meas_indices[b];

                        float cost_current = computeCostGLMB(ti_a, mi_a) + computeCostGLMB(ti_b, mi_b);
                        float cost_swapped = computeCostGLMB(ti_a, mi_b) + computeCostGLMB(ti_b, mi_a);

                        if (cost_swapped < cost_current * 0.95f) {
                            glmb_assoc_meas_indices[a] = mi_b;
                            glmb_assoc_meas_indices[b] = mi_a;
                            swapped = true;
                        }
                    }
                }
            }
        }

        // === GPU UKF バッチ更新 ===
        if (!glmb_assoc_track_indices.empty()) {
            int num_assoc = static_cast<int>(glmb_assoc_track_indices.size());

            std::vector<StateVector> assoc_states(num_assoc);
            std::vector<StateCov> assoc_covs(num_assoc);
            std::vector<float> assoc_meas(num_assoc * MEAS_DIM);

            for (int j = 0; j < num_assoc; j++) {
                int ti = glmb_assoc_track_indices[j];
                int mi = glmb_assoc_meas_indices[j];
                assoc_states[j] = tracks[ti].state;
                assoc_covs[j] = tracks[ti].covariance;

                assoc_meas[j * MEAS_DIM + 0] = measurements[mi].range;
                assoc_meas[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                assoc_meas[j * MEAS_DIM + 2] = measurements[mi].elevation;
                assoc_meas[j * MEAS_DIM + 3] = measurements[mi].doppler;
            }

            ukf_->copyToDevice(assoc_states, assoc_covs);
            d_meas_.copyFrom(assoc_meas.data(), static_cast<size_t>(num_assoc) * MEAS_DIM);
            ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                         d_meas_.get(), num_assoc, sensor_x_, sensor_y_, sensor_z_);
            ukf_->copyToHost(assoc_states, assoc_covs, num_assoc);

            // === IMM モデル確率更新 ===
            if (use_imm_ && imm_cpu_) {
                std::vector<int> imm_track_indices;
                std::vector<Measurement> imm_measurements;
                for (int j = 0; j < num_assoc; j++) {
                    int mi = glmb_assoc_meas_indices[j];
                    if (mi >= 0) {
                        imm_track_indices.push_back(glmb_assoc_track_indices[j]);
                        imm_measurements.push_back(measurements[mi]);
                    }
                }

                if (!imm_track_indices.empty()) {
                    std::vector<float> all_model_probs(tracks.size() * 4);
                    for (size_t ti = 0; ti < tracks.size(); ti++) {
                        if (tracks[ti].model_probs.size() == 4) {
                            all_model_probs[ti * 4 + 0] = tracks[ti].model_probs[0];
                            all_model_probs[ti * 4 + 1] = tracks[ti].model_probs[1];
                            all_model_probs[ti * 4 + 2] = tracks[ti].model_probs[2];
                            all_model_probs[ti * 4 + 3] = tracks[ti].model_probs[3];
                        } else {
                            all_model_probs[ti * 4 + 0] = 0.25f;
                            all_model_probs[ti * 4 + 1] = 0.25f;
                            all_model_probs[ti * 4 + 2] = 0.25f;
                            all_model_probs[ti * 4 + 3] = 0.25f;
                        }
                    }

                    imm_cpu_->updateModelProbabilities(
                        imm_track_indices, imm_measurements,
                        all_model_probs, sensor_x_, sensor_y_, sensor_z_);

                    for (size_t j = 0; j < imm_track_indices.size(); j++) {
                        int ti = imm_track_indices[j];
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.model_probs = {
                            all_model_probs[ti * 4 + 0],
                            all_model_probs[ti * 4 + 1],
                            all_model_probs[ti * 4 + 2],
                            all_model_probs[ti * 4 + 3]
                        };
                    }
                }
            }

            // UKF更新結果をトラックマネージャに反映
            for (int j = 0; j < num_assoc; j++) {
                int ti = glmb_assoc_track_indices[j];
                if (isStateValid(assoc_states[j], assoc_covs[j])) {
                    float dx = assoc_states[j](0) - tracks[ti].state(0);
                    float dy = assoc_states[j](1) - tracks[ti].state(1);
                    float dz = assoc_states[j](2) - tracks[ti].state(2);
                    float jump_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    float max_jump_vel = assoc_params_.max_jump_velocity;
                    float max_jump = (max_jump_vel > 0.0f)
                        ? std::max(max_jump_vel * static_cast<float>(dt) * 5.0f, 10000.0f)
                        : std::numeric_limits<float>::max();

                    if (jump_dist > max_jump) {
                        track_manager_->predictOnlyTrack(tracks[ti].id,
                                                         tracks[ti].state,
                                                         tracks[ti].covariance,
                                                         current_time);
                    } else {
                        int best_mi = glmb_assoc_meas_indices[j];
                        if (best_mi >= 0) {
                            Track& pre_track = track_manager_->getTrackMutable(tracks[ti].id);
                            pre_track.snr_sum += measurements[best_mi].snr;
                        }

                        track_manager_->updateTrack(tracks[ti].id,
                                                    assoc_states[j], assoc_covs[j],
                                                    current_time);
                    }
                } else {
                    track_manager_->predictOnlyTrack(tracks[ti].id,
                                                     tracks[ti].state,
                                                     tracks[ti].covariance,
                                                     current_time);
                }
            }
        }

        // === GLMB: 存在確率write-back ===
        for (const auto& upd : glmb_result.track_updates) {
            int ti = upd.track_index;
            if (track_manager_->hasTrack(tracks[ti].id)) {
                Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                track.existence_prob = upd.existence_prob;
            }
        }

        // === GLMB 救済: 未割当航跡を未使用高SNR観測で更新 ===
        std::set<int> glmb_used_meas;
        for (int mi : glmb_assoc_meas_indices) {
            if (mi >= 0) glmb_used_meas.insert(mi);
        }

        std::vector<int> glmb_truly_unassoc;
        for (int ti : glmb_unassoc_track_indices) {
            const Track& t = tracks[ti];
            bool rescue_eligible = false;
            if (t.track_state == TrackState::CONFIRMED && t.misses >= 1) {
                rescue_eligible = true;
            } else if (t.track_state == TrackState::TENTATIVE &&
                       t.snr_sum / std::max(t.hits, 1) >= assoc_params_.min_snr_for_init + 5.0f) {
                rescue_eligible = true;
            }
            if (!rescue_eligible) {
                glmb_truly_unassoc.push_back(ti);
                continue;
            }

            float tdx = t.state(0) - sensor_x_;
            float tdy = t.state(1) - sensor_y_;
            float tdz = t.state(2) - sensor_z_;
            float t_range = std::sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
            float t_az = std::atan2(tdy, tdx);
            float t_el_horiz = std::sqrt(tdx*tdx + tdy*tdy);
            float t_el = (t_el_horiz > 1e-6f) ? std::atan2(tdz, t_el_horiz) : 0.0f;

            int best_mi = -1;
            float best_cost = 25.0f;
            float rescue_snr_threshold = assoc_params_.min_snr_for_init + 2.0f;
            for (int mi : glmb_result.base_result.unassigned_measurements) {
                if (glmb_used_meas.count(mi)) continue;
                if (measurements[mi].snr < rescue_snr_threshold) continue;

                float dr = measurements[mi].range - t_range;
                float daz = measurements[mi].azimuth - t_az;
                if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
                if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
                float del = measurements[mi].elevation - t_el;

                float cost = std::sqrt(
                    (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                    (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                    (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise)
                );
                if (cost < best_cost) {
                    best_cost = cost;
                    best_mi = mi;
                }
            }

            if (best_mi >= 0) {
                StateVector rescue_state = t.state;
                StateCov rescue_cov = t.covariance;
                std::vector<StateVector> r_states = {rescue_state};
                std::vector<StateCov> r_covs = {rescue_cov};
                std::vector<float> r_meas = {
                    measurements[best_mi].range,
                    measurements[best_mi].azimuth,
                    measurements[best_mi].elevation,
                    measurements[best_mi].doppler
                };

                ukf_->copyToDevice(r_states, r_covs);
                d_meas_.copyFrom(r_meas.data(), MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), 1, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(r_states, r_covs, 1);

                if (isStateValid(r_states[0], r_covs[0])) {
                    Track& pre_track = track_manager_->getTrackMutable(t.id);
                    pre_track.snr_sum += measurements[best_mi].snr;
                    track_manager_->updateTrack(t.id, r_states[0], r_covs[0], current_time);
                    glmb_used_meas.insert(best_mi);
                } else {
                    glmb_truly_unassoc.push_back(ti);
                }
            } else {
                glmb_truly_unassoc.push_back(ti);
            }
        }

        // 未割当トラック: 予測のみ
        for (int ti : glmb_truly_unassoc) {
            track_manager_->predictOnlyTrack(tracks[ti].id,
                                             tracks[ti].state,
                                             tracks[ti].covariance,
                                             current_time);
        }

        // === GLMB ソフト共有: CONFIRMED クラスタ航跡のドリフト防止 ===
        {
            const float SOFT_SHARE_CLUSTER_SQ = 10000.0f * 10000.0f;
            const float SOFT_SHARE_MEAS_SQ = 5000.0f * 5000.0f;
            const float BLEND_ALPHA = 0.2f;

            std::vector<int> soft_share_tracks;
            std::vector<int> soft_share_meas;

            for (int ti : glmb_truly_unassoc) {
                if (tracks[ti].track_state != TrackState::CONFIRMED) continue;
                if (tracks[ti].misses < 2) continue;
                if (tracks[ti].existence_prob < 0.3f) continue;

                bool in_cluster = false;
                for (size_t aj = 0; aj < glmb_assoc_track_indices.size(); aj++) {
                    int ati = glmb_assoc_track_indices[aj];
                    if (tracks[ati].track_state != TrackState::CONFIRMED) continue;
                    float dx = tracks[ti].state(0) - tracks[ati].state(0);
                    float dy = tracks[ti].state(1) - tracks[ati].state(1);
                    float dz = tracks[ti].state(2) - tracks[ati].state(2);
                    if (dx*dx + dy*dy + dz*dz < SOFT_SHARE_CLUSTER_SQ) {
                        in_cluster = true;
                        break;
                    }
                }
                if (!in_cluster) continue;

                int best_meas = -1;
                float best_dist_sq = SOFT_SHARE_MEAS_SQ;
                for (int mi : glmb_assoc_meas_indices) {
                    if (mi < 0) continue;
                    float r = measurements[mi].range;
                    float az = measurements[mi].azimuth;
                    float el = measurements[mi].elevation;
                    float r_h = r * std::cos(el);
                    float mx = r_h * std::cos(az) + sensor_x_;
                    float my = r_h * std::sin(az) + sensor_y_;
                    float mz = r * std::sin(el) + sensor_z_;

                    float dx = mx - tracks[ti].state(0);
                    float dy = my - tracks[ti].state(1);
                    float dz = mz - tracks[ti].state(2);
                    float dist_sq = dx*dx + dy*dy + dz*dz;

                    if (dist_sq < best_dist_sq) {
                        best_dist_sq = dist_sq;
                        best_meas = mi;
                    }
                }

                if (best_meas >= 0) {
                    soft_share_tracks.push_back(ti);
                    soft_share_meas.push_back(best_meas);
                }
            }

            if (!soft_share_tracks.empty()) {
                int num_soft = static_cast<int>(soft_share_tracks.size());
                std::vector<StateVector> soft_states(num_soft);
                std::vector<StateCov> soft_covs(num_soft);
                std::vector<float> soft_meas_data(num_soft * MEAS_DIM);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    int mi = soft_share_meas[j];
                    soft_states[j] = tracks[ti].state;
                    soft_covs[j] = tracks[ti].covariance;
                    soft_meas_data[j * MEAS_DIM + 0] = measurements[mi].range;
                    soft_meas_data[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                    soft_meas_data[j * MEAS_DIM + 2] = measurements[mi].elevation;
                    soft_meas_data[j * MEAS_DIM + 3] = measurements[mi].doppler;
                }

                ukf_->copyToDevice(soft_states, soft_covs);
                d_meas_.copyFrom(soft_meas_data.data(), static_cast<size_t>(num_soft) * MEAS_DIM);
                ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                             d_meas_.get(), num_soft, sensor_x_, sensor_y_, sensor_z_);
                ukf_->copyToHost(soft_states, soft_covs, num_soft);

                for (int j = 0; j < num_soft; j++) {
                    int ti = soft_share_tracks[j];
                    if (isStateValid(soft_states[j], soft_covs[j])) {
                        Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                        track.state = (1.0f - BLEND_ALPHA) * tracks[ti].state
                                    + BLEND_ALPHA * soft_states[j];
                        if (track.hits >= 10 && track.misses > 0) {
                            track.misses--;
                        }
                    }
                }
            }
        }

        // 新規トラック初期化
        {
            std::vector<int> glmb_final_unassigned_meas;
            for (int mi : glmb_result.base_result.unassigned_measurements) {
                if (glmb_used_meas.count(mi)) continue;
                glmb_final_unassigned_meas.push_back(mi);
            }
            initializeNewTracks(glmb_final_unassigned_meas, measurements);
        }

        // 新規トラックに初期存在確率を設定
        {
            auto all_tracks_now = track_manager_->getAllTracks();
            for (const auto& t : all_tracks_now) {
                if (t.age == 0 && t.track_state == TrackState::TENTATIVE) {
                    Track& mt = track_manager_->getTrackMutable(t.id);
                    mt.existence_prob = assoc_params_.glmb_initial_existence;
                }
            }
        }

        pruneTracks();

        auto update_end_glmb = std::chrono::high_resolution_clock::now();

        last_perf_stats_.predict_time_ms = std::chrono::duration<double, std::milli>(
            predict_end - predict_start).count();
        last_perf_stats_.association_time_ms = std::chrono::duration<double, std::milli>(
            assoc_end_glmb - assoc_start).count();
        last_perf_stats_.update_time_ms = std::chrono::duration<double, std::milli>(
            update_end_glmb - update_start_glmb).count();
        last_perf_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            update_end_glmb - start_time).count();
        last_perf_stats_.num_tracks = static_cast<int>(tracks.size());
        last_perf_stats_.num_measurements = static_cast<int>(measurements.size());

        last_update_time_ = current_time;
        total_updates_++;
        total_processing_time_ += last_perf_stats_.total_time_ms;
        total_measurements_processed_ += static_cast<int>(measurements.size());
        return;
    }

    // ========================================
    // GNN モード（既存フロー）
    // ========================================
    AssociationResult assoc_result = data_association_->associate(tracks, measurements);
    auto assoc_end = std::chrono::high_resolution_clock::now();

    // === 更新ステップ ===
    auto update_start = std::chrono::high_resolution_clock::now();

    // === 近接CONFIRMED航跡のGNN割当局所補正 (2-opt swap) ===
    // GNNのHungarian法はグローバル最適だが、近接航跡の局所割当が最適でない場合がある。
    // 2つの近接CONFIRMED航跡の割当観測を交換した方がペアコストが改善する場合、交換する。
    {
        const float CLUSTER_DIST_SQ = 5000.0f * 5000.0f;  // 5km

        // 割当済みCONFIRMED航跡のリスト
        struct AssocEntry { int track_idx; int meas_idx; };
        std::vector<AssocEntry> confirmed_assoc;
        for (size_t i = 0; i < tracks.size(); i++) {
            int mi = assoc_result.track_to_meas[i];
            if (mi >= 0 && tracks[i].track_state == TrackState::CONFIRMED) {
                confirmed_assoc.push_back({static_cast<int>(i), mi});
            }
        }

        // 観測値→測定空間変換関数（正規化距離計算用）
        auto computeCost = [&](int ti, int mi) -> float {
            float dx = tracks[ti].state(0) - sensor_x_;
            float dy = tracks[ti].state(1) - sensor_y_;
            float dz = tracks[ti].state(2) - sensor_z_;
            float pred_range = std::sqrt(dx*dx + dy*dy + dz*dz);
            float pred_az = std::atan2(dy, dx);
            float rh = std::sqrt(dx*dx + dy*dy);
            float pred_el = (rh > 1e-6f) ? std::atan2(dz, rh) : 0.0f;

            float dr = measurements[mi].range - pred_range;
            float daz = measurements[mi].azimuth - pred_az;
            if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
            if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
            float del = measurements[mi].elevation - pred_el;

            return (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                   (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                   (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise);
        };

        // 2-opt: 近接ペアで割当交換が改善するか判定
        bool swapped = true;
        while (swapped) {
            swapped = false;
            for (size_t a = 0; a < confirmed_assoc.size(); a++) {
                for (size_t b = a + 1; b < confirmed_assoc.size(); b++) {
                    int ti_a = confirmed_assoc[a].track_idx;
                    int ti_b = confirmed_assoc[b].track_idx;

                    // 近接チェック
                    float ddx = tracks[ti_a].state(0) - tracks[ti_b].state(0);
                    float ddy = tracks[ti_a].state(1) - tracks[ti_b].state(1);
                    float ddz = tracks[ti_a].state(2) - tracks[ti_b].state(2);
                    if (ddx*ddx + ddy*ddy + ddz*ddz >= CLUSTER_DIST_SQ) continue;

                    int mi_a = confirmed_assoc[a].meas_idx;
                    int mi_b = confirmed_assoc[b].meas_idx;

                    // 現在のコスト vs 交換後のコスト
                    float cost_current = computeCost(ti_a, mi_a) + computeCost(ti_b, mi_b);
                    float cost_swapped = computeCost(ti_a, mi_b) + computeCost(ti_b, mi_a);

                    if (cost_swapped < cost_current * 0.95f) {  // 5%以上改善時のみ交換
                        assoc_result.track_to_meas[ti_a] = mi_b;
                        assoc_result.track_to_meas[ti_b] = mi_a;
                        assoc_result.meas_to_track[mi_a] = ti_b;
                        assoc_result.meas_to_track[mi_b] = ti_a;
                        confirmed_assoc[a].meas_idx = mi_b;
                        confirmed_assoc[b].meas_idx = mi_a;
                        swapped = true;
                    }
                }
            }
        }
    }

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
            std::vector<float> all_model_probs(tracks.size() * 4);
            for (size_t ti = 0; ti < tracks.size(); ti++) {
                if (tracks[ti].model_probs.size() == 4) {
                    all_model_probs[ti * 4 + 0] = tracks[ti].model_probs[0];
                    all_model_probs[ti * 4 + 1] = tracks[ti].model_probs[1];
                    all_model_probs[ti * 4 + 2] = tracks[ti].model_probs[2];
                    all_model_probs[ti * 4 + 3] = tracks[ti].model_probs[3];
                } else {
                    all_model_probs[ti * 4 + 0] = 0.25f;
                    all_model_probs[ti * 4 + 1] = 0.25f;
                    all_model_probs[ti * 4 + 2] = 0.25f;
                    all_model_probs[ti * 4 + 3] = 0.25f;
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
                    all_model_probs[ti * 4 + 0],
                    all_model_probs[ti * 4 + 1],
                    all_model_probs[ti * 4 + 2],
                    all_model_probs[ti * 4 + 3]
                };
            }
        }

        // トラックマネージャに反映
        for (int j = 0; j < num_assoc; j++) {
            int ti = associated_track_indices[j];
            if (isStateValid(assoc_states[j], assoc_covs[j])) {
                // 位置ジャンプチェック: 予測→更新の移動距離が物理上限を超えたら棄却
                float dx = assoc_states[j](0) - tracks[ti].state(0);
                float dy = assoc_states[j](1) - tracks[ti].state(1);
                float dz = assoc_states[j](2) - tracks[ti].state(2);
                float jump_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                // 最大許容ジャンプ: max_jump_velocity × dt × 5倍安全率、最低10km (0=無制限)
                float max_jump_vel = assoc_params_.max_jump_velocity;
                float max_jump = (max_jump_vel > 0.0f)
                    ? std::max(max_jump_vel * static_cast<float>(dt) * 5.0f, 10000.0f)
                    : std::numeric_limits<float>::max();

                if (jump_dist > max_jump) {
                    std::cerr << "  [JUMP REJECT] Track " << tracks[ti].id
                              << " jump=" << (jump_dist/1000.0f) << " km"
                              << " > max=" << (max_jump/1000.0f) << " km"
                              << " -> predict only" << std::endl;
                    track_manager_->predictOnlyTrack(tracks[ti].id,
                                                     tracks[ti].state,
                                                     tracks[ti].covariance,
                                                     current_time);
                    // 棄却された観測値を未割り当てに戻す（新規トラック初期化に利用可能）
                    assoc_result.unassigned_measurements.push_back(associated_meas_indices[j]);
                } else {
                    // SNR蓄積を更新前に実行（updateTrackState内の確認チェックで全ヒットのSNRが必要）
                    int mi = associated_meas_indices[j];
                    Track& pre_track = track_manager_->getTrackMutable(tracks[ti].id);
                    pre_track.snr_sum += measurements[mi].snr;

                    track_manager_->updateTrack(tracks[ti].id,
                                                assoc_states[j], assoc_covs[j],
                                                current_time);
                }
            } else {
                // UKF更新結果が異常値 → 今フレームの予測状態を維持してミス扱い
                track_manager_->predictOnlyTrack(tracks[ti].id,
                                                 tracks[ti].state,
                                                 tracks[ti].covariance,
                                                 current_time);
                // 異常値の観測値も未割り当てに戻す
                assoc_result.unassigned_measurements.push_back(associated_meas_indices[j]);
            }
        }
    }

    // === 観測値共有（簡易JPDA）: 未分解目標の航跡継続を支援 ===
    // 観測なしの航跡に対し、近傍の既割り当て観測値を共有してUKF更新する。
    //
    // 重要: CONFIRMED航跡間での観測共有は無効化する。
    // 理由: 別目標の観測でUKF更新すると、航跡の状態推定が汚染され、
    //       実目標位置からドリフトし、ビーム照射位置が不正確になり、
    //       連続BeamMissを招いて結果的にtrack deathが加速する。
    // CONFIRMED航跡は観測なしでもコースティングで生存し、adaptive delete_misses
    // で十分な猶予を持つため、誤った観測共有より安全。
    // （近接500m以内のCONFIRMED共有は上記で処理済み）
    const int SHARING_AGE_LIMIT = 30;  // TENTATIVE航跡の年齢制限
    const float SHARING_DIST_SQ = 5000.0f * 5000.0f;  // 5km以内の観測を共有

    std::vector<int> shared_track_indices;
    std::vector<int> shared_meas_indices;
    std::vector<int> remaining_unassoc_tracks;

    for (int ti : unassociated_track_indices) {
        const Track& unassigned = tracks[ti];
        // 共有条件: TENTATIVEのみ（若い航跡で、少なくとも1回は検出済み）
        // CONFIRMED航跡は対象外（別目標の観測による状態汚染を防止）
        bool eligible = false;
        if (unassigned.track_state == TrackState::TENTATIVE
            && unassigned.age <= SHARING_AGE_LIMIT
            && unassigned.hits > 0) {
            eligible = true;
        }

        if (eligible && !associated_meas_indices.empty()) {
            // 最近傍の割り当て済み観測を検索（直交座標距離）
            int best_meas = -1;
            float best_dist_sq = SHARING_DIST_SQ;

            for (size_t j = 0; j < associated_meas_indices.size(); j++) {
                int mi = associated_meas_indices[j];
                float r = measurements[mi].range;
                float az = measurements[mi].azimuth;
                float el = measurements[mi].elevation;
                float r_h = r * std::cos(el);
                float mx = r_h * std::cos(az) + sensor_x_;
                float my = r_h * std::sin(az) + sensor_y_;
                float mz = r * std::sin(el) + sensor_z_;

                float dx = mx - unassigned.state(0);
                float dy = my - unassigned.state(1);
                float dz = mz - unassigned.state(2);
                float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_meas = mi;
                }
            }

            if (best_meas >= 0) {
                shared_track_indices.push_back(ti);
                shared_meas_indices.push_back(best_meas);
                continue;
            }
        }
        remaining_unassoc_tracks.push_back(ti);
    }

    // 共有航跡のバッチUKF更新
    if (!shared_track_indices.empty()) {
        int num_shared = static_cast<int>(shared_track_indices.size());
        std::vector<StateVector> shared_states(num_shared);
        std::vector<StateCov> shared_covs(num_shared);
        std::vector<float> shared_meas(num_shared * MEAS_DIM);

        for (int j = 0; j < num_shared; j++) {
            int ti = shared_track_indices[j];
            int mi = shared_meas_indices[j];
            shared_states[j] = tracks[ti].state;
            shared_covs[j] = tracks[ti].covariance;
            shared_meas[j * MEAS_DIM + 0] = measurements[mi].range;
            shared_meas[j * MEAS_DIM + 1] = measurements[mi].azimuth;
            shared_meas[j * MEAS_DIM + 2] = measurements[mi].elevation;
            shared_meas[j * MEAS_DIM + 3] = measurements[mi].doppler;
        }

        ukf_->copyToDevice(shared_states, shared_covs);
        d_meas_.copyFrom(shared_meas.data(), static_cast<size_t>(num_shared) * MEAS_DIM);
        ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                     d_meas_.get(), num_shared, sensor_x_, sensor_y_, sensor_z_);
        ukf_->copyToHost(shared_states, shared_covs, num_shared);

        for (int j = 0; j < num_shared; j++) {
            int ti = shared_track_indices[j];
            if (isStateValid(shared_states[j], shared_covs[j])) {
                track_manager_->updateTrack(tracks[ti].id,
                                            shared_states[j], shared_covs[j],
                                            current_time);
            } else {
                track_manager_->predictOnlyTrack(tracks[ti].id,
                                                 tracks[ti].state,
                                                 tracks[ti].covariance,
                                                 current_time);
                remaining_unassoc_tracks.push_back(ti);
            }
        }
    }

    // === 救済アソシエーション: Hungarian漏れのTENTATIVE航跡を最近傍観測で救済 ===
    // Hungarian法は正規化距離コスト行列で最適割当を行うが、TENTATIVE航跡の予測誤差が
    // 大きい場合、真目標の観測がクラッタ航跡に割当てられ真目標TENTATIVEが未割当になる。
    // この救済パスは未割当TENTATIVEに対し、未使用の高SNR観測を最近傍で割当てる。
    std::set<int> used_meas_set(associated_meas_indices.begin(), associated_meas_indices.end());
    // 共有で使用した観測も除外
    for (int mi : shared_meas_indices) {
        used_meas_set.insert(mi);
    }

    std::vector<int> final_unassoc_tracks;
    for (int ti : remaining_unassoc_tracks) {
        const Track& t = tracks[ti];
        // 救済対象: 高SNR TENTATIVE航跡のみ（真目標候補のみ救済、クラッタ排除）
        if (t.track_state != TrackState::TENTATIVE ||
            t.snr_sum / std::max(t.hits, 1) < assoc_params_.min_snr_for_init + 5.0f) {
            final_unassoc_tracks.push_back(ti);
            continue;
        }

        // TENTATIVE航跡の予測観測を計算（センサー基準）
        float tdx = t.state(0) - sensor_x_;
        float tdy = t.state(1) - sensor_y_;
        float tdz = t.state(2) - sensor_z_;
        float t_range = std::sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
        float t_az = std::atan2(tdy, tdx);
        float t_el_horiz = std::sqrt(tdx*tdx + tdy*tdy);
        float t_el = (t_el_horiz > 1e-6f) ? std::atan2(tdz, t_el_horiz) : 0.0f;

        // 最近傍の未使用高SNR観測を検索（厳密ゲート＋高SNR要件）
        int best_mi = -1;
        float best_cost = 15.0f;  // 厳密ゲート（正規化距離15以内 ≈ 150m range）
        float rescue_snr_threshold = assoc_params_.min_snr_for_init + 8.0f;  // 18dB
        for (int mi : assoc_result.unassigned_measurements) {
            if (used_meas_set.count(mi)) continue;
            if (measurements[mi].snr < rescue_snr_threshold) continue;

            float dr = measurements[mi].range - t_range;
            float daz = measurements[mi].azimuth - t_az;
            if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
            if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
            float del = measurements[mi].elevation - t_el;

            float cost = std::sqrt(
                (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise)
            );
            if (cost < best_cost) {
                best_cost = cost;
                best_mi = mi;
            }
        }

        if (best_mi >= 0) {
            // UKF更新を実行
            StateVector rescue_state = t.state;
            StateCov rescue_cov = t.covariance;
            std::vector<StateVector> r_states = {rescue_state};
            std::vector<StateCov> r_covs = {rescue_cov};
            std::vector<float> r_meas = {
                measurements[best_mi].range,
                measurements[best_mi].azimuth,
                measurements[best_mi].elevation,
                measurements[best_mi].doppler
            };

            ukf_->copyToDevice(r_states, r_covs);
            d_meas_.copyFrom(r_meas.data(), MEAS_DIM);
            ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                         d_meas_.get(), 1, sensor_x_, sensor_y_, sensor_z_);
            ukf_->copyToHost(r_states, r_covs, 1);

            if (isStateValid(r_states[0], r_covs[0])) {
                // SNR蓄積 + 更新
                Track& pre_track = track_manager_->getTrackMutable(t.id);
                pre_track.snr_sum += measurements[best_mi].snr;
                track_manager_->updateTrack(t.id, r_states[0], r_covs[0], current_time);
                used_meas_set.insert(best_mi);
            } else {
                final_unassoc_tracks.push_back(ti);
            }
        } else {
            final_unassoc_tracks.push_back(ti);
        }
    }

    // === CONFIRMED救済: 未割当CONFIRMED航跡を未使用高SNR観測で更新 ===
    // GNN飢餓で観測を失ったCONFIRMED航跡に対し、未使用の高SNR観測を
    // 最近傍マッチングで割当てる。これによりミスカウンタがリセットされ、
    // 航跡のLOST遷移を防止する。TENTATIVE救済と同様のゲート判定を使用。
    std::vector<int> truly_unassoc_tracks;
    for (int ti : final_unassoc_tracks) {
        const Track& t = tracks[ti];
        if (t.track_state != TrackState::CONFIRMED || t.misses < 1) {
            truly_unassoc_tracks.push_back(ti);
            continue;
        }

        // CONFIRMED航跡の予測観測を計算
        float tdx = t.state(0) - sensor_x_;
        float tdy = t.state(1) - sensor_y_;
        float tdz = t.state(2) - sensor_z_;
        float t_range = std::sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
        float t_az = std::atan2(tdy, tdx);
        float t_el_horiz = std::sqrt(tdx*tdx + tdy*tdy);
        float t_el = (t_el_horiz > 1e-6f) ? std::atan2(tdz, t_el_horiz) : 0.0f;

        // 未使用観測を検索（緩和ゲート内最近傍）
        int best_mi = -1;
        float best_cost = 25.0f;  // 緩和ゲート（CONFIRMED航跡の予測誤差が大きい場合に対応）
        float rescue_snr_threshold = assoc_params_.min_snr_for_init + 2.0f;  // 12dB
        for (int mi : assoc_result.unassigned_measurements) {
            if (used_meas_set.count(mi)) continue;
            if (measurements[mi].snr < rescue_snr_threshold) continue;

            float dr = measurements[mi].range - t_range;
            float daz = measurements[mi].azimuth - t_az;
            if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
            if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
            float del = measurements[mi].elevation - t_el;

            float cost = std::sqrt(
                (dr / meas_noise_.range_noise) * (dr / meas_noise_.range_noise) +
                (daz / meas_noise_.azimuth_noise) * (daz / meas_noise_.azimuth_noise) +
                (del / meas_noise_.elevation_noise) * (del / meas_noise_.elevation_noise)
            );
            if (cost < best_cost) {
                best_cost = cost;
                best_mi = mi;
            }
        }

        if (best_mi >= 0) {
            // UKF更新を実行
            StateVector rescue_state = t.state;
            StateCov rescue_cov = t.covariance;
            std::vector<StateVector> r_states = {rescue_state};
            std::vector<StateCov> r_covs = {rescue_cov};
            std::vector<float> r_meas = {
                measurements[best_mi].range,
                measurements[best_mi].azimuth,
                measurements[best_mi].elevation,
                measurements[best_mi].doppler
            };

            ukf_->copyToDevice(r_states, r_covs);
            d_meas_.copyFrom(r_meas.data(), MEAS_DIM);
            ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                         d_meas_.get(), 1, sensor_x_, sensor_y_, sensor_z_);
            ukf_->copyToHost(r_states, r_covs, 1);

            if (isStateValid(r_states[0], r_covs[0])) {
                // SNR蓄積 + updateTrack（ヒットとしてカウント）
                Track& pre_track = track_manager_->getTrackMutable(t.id);
                pre_track.snr_sum += measurements[best_mi].snr;
                track_manager_->updateTrack(t.id, r_states[0], r_covs[0], current_time);
                used_meas_set.insert(best_mi);
            } else {
                truly_unassoc_tracks.push_back(ti);
            }
        } else {
            truly_unassoc_tracks.push_back(ti);
        }
    }

    // 観測が割り当てられなかったトラック: 予測のみ
    for (int ti : truly_unassoc_tracks) {
        track_manager_->predictOnlyTrack(tracks[ti].id,
                                         tracks[ti].state,
                                         tracks[ti].covariance,
                                         current_time);
    }

    // === ソフト共有: CONFIRMED クラスタ航跡のドリフト防止 ===
    // GNN飢餓で連続ミス中のCONFIRMED航跡は予測のみで位置が真値からドリフトし、
    // コスト行列で不利→さらに飢餓、のフィードバックループが発生する。
    // 近傍の既割当観測でUKF更新し、結果を20%だけブレンドして軽微な位置補正を行う。
    // ミスカウンタは上記predictOnlyTrackで既にインクリメント済み（状態遷移は変更しない）。
    {
        const float SOFT_SHARE_CLUSTER_SQ = 10000.0f * 10000.0f;  // 10km
        const float SOFT_SHARE_MEAS_SQ = 5000.0f * 5000.0f;       // 5km
        const float BLEND_ALPHA = 0.2f;  // 20% measurement, 80% prediction

        std::vector<int> soft_share_tracks;
        std::vector<int> soft_share_meas;

        for (int ti : truly_unassoc_tracks) {
            if (tracks[ti].track_state != TrackState::CONFIRMED) continue;
            if (tracks[ti].misses < 2) continue;

            // クラスタ内かチェック（近傍に割当済みCONFIRMED航跡があるか）
            bool in_cluster = false;
            for (size_t j = 0; j < associated_track_indices.size(); j++) {
                int aj = associated_track_indices[j];
                if (tracks[aj].track_state != TrackState::CONFIRMED) continue;
                float dx = tracks[ti].state(0) - tracks[aj].state(0);
                float dy = tracks[ti].state(1) - tracks[aj].state(1);
                float dz = tracks[ti].state(2) - tracks[aj].state(2);
                if (dx*dx + dy*dy + dz*dz < SOFT_SHARE_CLUSTER_SQ) {
                    in_cluster = true;
                    break;
                }
            }
            if (!in_cluster) continue;

            // 最近傍の割当済み観測を検索
            int best_meas = -1;
            float best_dist_sq = SOFT_SHARE_MEAS_SQ;
            for (size_t j = 0; j < associated_meas_indices.size(); j++) {
                int mi = associated_meas_indices[j];
                float r = measurements[mi].range;
                float az = measurements[mi].azimuth;
                float el = measurements[mi].elevation;
                float r_h = r * std::cos(el);
                float mx = r_h * std::cos(az) + sensor_x_;
                float my = r_h * std::sin(az) + sensor_y_;
                float mz = r * std::sin(el) + sensor_z_;

                float dx = mx - tracks[ti].state(0);
                float dy = my - tracks[ti].state(1);
                float dz = mz - tracks[ti].state(2);
                float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_meas = mi;
                }
            }

            if (best_meas >= 0) {
                soft_share_tracks.push_back(ti);
                soft_share_meas.push_back(best_meas);
            }
        }

        // バッチUKF更新 + ブレンド
        if (!soft_share_tracks.empty()) {
            int num_soft = static_cast<int>(soft_share_tracks.size());
            std::vector<StateVector> soft_states(num_soft);
            std::vector<StateCov> soft_covs(num_soft);
            std::vector<float> soft_meas_data(num_soft * MEAS_DIM);

            for (int j = 0; j < num_soft; j++) {
                int ti = soft_share_tracks[j];
                int mi = soft_share_meas[j];
                soft_states[j] = tracks[ti].state;
                soft_covs[j] = tracks[ti].covariance;
                soft_meas_data[j * MEAS_DIM + 0] = measurements[mi].range;
                soft_meas_data[j * MEAS_DIM + 1] = measurements[mi].azimuth;
                soft_meas_data[j * MEAS_DIM + 2] = measurements[mi].elevation;
                soft_meas_data[j * MEAS_DIM + 3] = measurements[mi].doppler;
            }

            ukf_->copyToDevice(soft_states, soft_covs);
            d_meas_.copyFrom(soft_meas_data.data(), static_cast<size_t>(num_soft) * MEAS_DIM);
            ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                         d_meas_.get(), num_soft, sensor_x_, sensor_y_, sensor_z_);
            ukf_->copyToHost(soft_states, soft_covs, num_soft);

            for (int j = 0; j < num_soft; j++) {
                int ti = soft_share_tracks[j];
                if (isStateValid(soft_states[j], soft_covs[j])) {
                    // ブレンド: 予測状態を80%維持、UKF更新結果を20%混合
                    Track& track = track_manager_->getTrackMutable(tracks[ti].id);
                    track.state = (1.0f - BLEND_ALPHA) * tracks[ti].state
                                + BLEND_ALPHA * soft_states[j];
                    // 共分散は予測値を維持（ブレンドすると過剰に縮小する恐れ）

                    // ミスカウンタ半減: ソフト共有で部分観測があるため、
                    // 完全ミスとしてカウントしない。hits>=10の確立済み航跡のみ。
                    // これによりGNN飢餓中のCONFIRMED航跡の生存期間が延長される。
                    if (track.hits >= 10 && track.misses > 0) {
                        track.misses--;  // predictOnlyTrackで増えた1をキャンセル
                    }
                }
            }
        }
    }

    // 未使用の観測を新規トラック初期化用に更新
    std::vector<int> final_unassigned_meas;
    for (int mi : assoc_result.unassigned_measurements) {
        if (!used_meas_set.count(mi)) {
            final_unassigned_meas.push_back(mi);
        }
    }

    // 新規トラック初期化
    initializeNewTracks(final_unassigned_meas, measurements);

    // === CONFIRMED航跡収束プルーニング（無効化） ===
    // 収束プルーニングを無効化: 一時的に近接した別目標航跡の誤削除を防止。
    // 近接目標ペア（分離後の弾頭/ブースター）は測定空間で3σ以内に収束するが、
    // 同一目標ではないため、プルーニングすると有用な航跡が失われる。
    // プルーニングなしでMT=2が0/10に改善（全シードMT=3+）。
    // pruneConvergedTracks();

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
        if (use_imm_ && track.model_probs.size() == 4) {
            for (float prob : track.model_probs) {
                model_probs.push_back(prob);
            }
        } else if (use_imm_) {
            // デフォルト均等確率
            model_probs.push_back(0.25f);
            model_probs.push_back(0.25f);
            model_probs.push_back(0.25f);
            model_probs.push_back(0.25f);
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
            if (updated_probs.size() >= (i + 1) * 4) {
                track.model_probs.clear();
                track.model_probs.push_back(updated_probs[i * 4]);
                track.model_probs.push_back(updated_probs[i * 4 + 1]);
                track.model_probs.push_back(updated_probs[i * 4 + 2]);
                track.model_probs.push_back(updated_probs[i * 4 + 3]);
            }

            // コースト中のモデル確率均一化: 連続ミスが続くと観測尤度による
            // モデル更新が行われず、最後の観測時点の確率に固着する。
            // 例: BAL=0.46で固着→予測が弾道降下一辺倒→HGV上昇を追従不能。
            // 均一方向にブレンドすることでCTモデル（高プロセスノイズ）の
            // 影響力を回復し、予測共分散の成長を促進する。
            if (tracks[i].misses > 5 && track.model_probs.size() == 4) {
                float blend = std::min(0.05f * static_cast<float>(tracks[i].misses - 5), 0.4f);
                float sum = 0.0f;
                for (size_t j = 0; j < 4; j++) {
                    track.model_probs[j] = (1.0f - blend) * track.model_probs[j] + blend / 4.0f;
                    sum += track.model_probs[j];
                }
                if (sum > 0) {
                    for (size_t j = 0; j < 4; j++) track.model_probs[j] /= sum;
                }
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

    // コースト膨張: 連続ミス中の航跡の共分散を膨張させ、ゲートを広く保つ
    // これにより一時的な検出途切れ後の再捕捉確率が向上する
    //
    // 近接CONFIRMED航跡がある場合は膨張を抑制する。
    // 理由: ゲートを広げすぎると隣接目標の観測を誤取得し、
    //       状態推定が汚染されてBeamMissが連鎖する。
    for (const auto& orig_track : tracks) {
        if (orig_track.misses > 0) {
            Track& track = track_manager_->getTrackMutable(orig_track.id);

            // 近接CONFIRMED航跡チェック
            bool has_nearby_confirmed = false;
            if (orig_track.track_state == TrackState::CONFIRMED) {
                float prox_sq = 10000.0f * 10000.0f;  // 10km
                for (const auto& other : tracks) {
                    if (other.id == orig_track.id) continue;
                    if (other.track_state == TrackState::CONFIRMED) {
                        float dx = orig_track.state(0) - other.state(0);
                        float dy = orig_track.state(1) - other.state(1);
                        float dz = orig_track.state(2) - other.state(2);
                        if (dx*dx + dy*dy + dz*dz < prox_sq) {
                            has_nearby_confirmed = true;
                            break;
                        }
                    }
                }
            }

            float coast_factor;
            if (has_nearby_confirmed) {
                // 近接航跡あり: 線形膨張（穏やか）でゲートを狭く保つ
                coast_factor = 1.0f + 0.02f * static_cast<float>(orig_track.misses);
                coast_factor = std::min(coast_factor, 2.0f);
            } else {
                // 単独航跡: 二次的膨張で再捕捉確率を最大化
                coast_factor = 1.0f + 0.05f * static_cast<float>(orig_track.misses * orig_track.misses);
                coast_factor = std::min(coast_factor, 10.0f);
            }
            track.covariance *= coast_factor;

            // 垂直方向追加膨張: HGV機動は主に高度方向（z軸）で発生。
            // 水平方向は比較的予測通りだが、スキップ機動中の高度変化は
            // どのIMMモデル（CA/BAL/CT）も捕捉できない。
            // z/vz/az成分を追加膨張して垂直方向の不確かさを適切に反映。
            // 再検出時のフィルタ更新でカルマンゲインが観測重視になり、
            // 状態修正が高速化する。
            if (!has_nearby_confirmed && orig_track.misses >= 5) {
                float z_extra = 1.0f + 0.2f * static_cast<float>(orig_track.misses - 5);
                z_extra = std::min(z_extra, 5.0f);
                track.covariance(2, 2) *= z_extra;  // z position
                track.covariance(5, 5) *= z_extra;  // vz velocity
                track.covariance(8, 8) *= z_extra;  // az acceleration
            }
        }
    }
}

void MultiTargetTracker::initializeNewTracks(
    const std::vector<int>& unassigned_measurements,
    const std::vector<Measurement>& measurements)
{
    // 重複航跡防止リスト: 同一フレーム内の新規航跡 + 既存確認済み航跡
    std::vector<StateVector> new_track_states;

    // 既存航跡の位置を収集（重複生成防止）
    // 分離目標検出のための階層的距離閾値:
    //   CONFIRMED: 200m（分離目標は50m+で分解可能、200m安全マージン）
    //   TENTATIVE/LOST: 近接チェック対象外
    //     TENTATIVE: SNR初期化閾値で十分なフィルタリング済み。
    //       TENTATIVE近接ブロックはクラッタ起源の成熟TENTATIVEが真目標の
    //       TENTATIVE生成を阻害する副作用があるため無効化。
    //     LOST: 予測位置が不正確になるため新規生成を妨げない。
    struct ProximityEntry { StateVector state; float dist_sq; };
    std::vector<ProximityEntry> existing_track_entries;
    if (assoc_params_.min_init_distance > 0.0f) {
        float confirmed_dist_sq = 100.0f * 100.0f;  // 100m (分離後~1.5秒で超過)
        auto all_tracks = track_manager_->getAllTracks();
        for (const auto& t : all_tracks) {
            // 健全CONFIRMED航跡のみ: 瀕死航跡(misses>=3)は後継TENTATIVE生成をブロックしない
            if (t.track_state == TrackState::CONFIRMED && t.misses < 3) {
                existing_track_entries.push_back({t.state, confirmed_dist_sq});
            }
        }
    }

    for (int idx : unassigned_measurements) {
        // トラック数の上限チェック
        if (track_manager_->getNumTracks() >= max_targets_) {
            break;
        }

        const auto& meas = measurements[idx];

        // SNRベースのフィルタリング（TENTATIVE初期化用）
        float init_snr_threshold = assoc_params_.min_snr_for_init + 8.0f;
        if (meas.snr < init_snr_threshold) {
            continue;
        }

        // 既存航跡・同一フレーム新規航跡との距離チェック
        if (assoc_params_.min_init_distance > 0.0f) {
            float r = meas.range;
            float az = meas.azimuth;
            float el = meas.elevation;
            float r_horiz = r * std::cos(el);
            float mx = r_horiz * std::cos(az) + sensor_x_;
            float my = r_horiz * std::sin(az) + sensor_y_;
            float mz = r * std::sin(el) + sensor_z_;

            bool too_close = false;
            float full_dist_sq = assoc_params_.min_init_distance * assoc_params_.min_init_distance;
            for (const auto& entry : existing_track_entries) {
                float dx = mx - entry.state(0), dy = my - entry.state(1), dz = mz - entry.state(2);
                if (dx*dx + dy*dy + dz*dz < entry.dist_sq) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                for (const auto& s : new_track_states) {
                    float dx = mx - s(0), dy = my - s(1), dz = mz - s(2);
                    if (dx*dx + dy*dy + dz*dz < full_dist_sq) {
                        too_close = true;
                        break;
                    }
                }
            }
            if (too_close) { continue; }
        }

        int new_id = track_manager_->initializeTrack(meas);

        // 新規航跡の位置を重複防止リストに追加
        if (assoc_params_.min_init_distance > 0.0f) {
            auto all_tracks = track_manager_->getAllTracks();
            for (const auto& t : all_tracks) {
                if (t.id == new_id) {
                    new_track_states.push_back(t.state);
                    break;
                }
            }
        }
    }
}

void MultiTargetTracker::pruneTracks() {
    track_manager_->pruneLostTracks();
}

void MultiTargetTracker::pruneConvergedTracks() {
    // CONFIRMED航跡が測定空間で収束した場合（同一目標を追尾）、
    // 低品質側を削除して観測リソースを解放する。
    // これにより孤立した目標が新規トラックを獲得しやすくなる。
    //
    // 収束判定: 予測測定値のレンジ差 < 3σ_range AND 方位角差 < 3σ_azimuth
    // → 同一目標をレーダーが分解できない距離に収束
    auto confirmed = track_manager_->getConfirmedTracks();
    if (confirmed.size() < 2) return;

    float range_threshold = 3.0f * meas_noise_.range_noise;    // 30m
    float az_threshold = 3.0f * meas_noise_.azimuth_noise;      // 0.03 rad

    // 各CONFIRMED航跡の予測測定値を計算
    struct PredMeas {
        int track_id;
        float range;
        float azimuth;
        int hits;
    };
    std::vector<PredMeas> preds;
    preds.reserve(confirmed.size());

    for (const auto& t : confirmed) {
        float dx = t.state(0) - sensor_x_;
        float dy = t.state(1) - sensor_y_;
        float dz = t.state(2) - sensor_z_;
        float range = std::sqrt(dx*dx + dy*dy + dz*dz);
        float azimuth = std::atan2(dy, dx);
        preds.push_back({t.id, range, azimuth, t.hits});
    }

    // 収束ペア検出: 測定空間で近接するペアの低品質側を削除対象にする
    std::set<int> to_delete;
    for (size_t i = 0; i < preds.size(); i++) {
        if (to_delete.count(preds[i].track_id)) continue;
        for (size_t j = i + 1; j < preds.size(); j++) {
            if (to_delete.count(preds[j].track_id)) continue;

            float dr = std::abs(preds[i].range - preds[j].range);
            float daz = preds[i].azimuth - preds[j].azimuth;
            if (daz > static_cast<float>(M_PI)) daz -= 2.0f * static_cast<float>(M_PI);
            if (daz < -static_cast<float>(M_PI)) daz += 2.0f * static_cast<float>(M_PI);
            float daz_abs = std::abs(daz);

            if (dr < range_threshold && daz_abs < az_threshold) {
                // 収束検出: 低ヒット側を削除
                int victim_id = (preds[i].hits <= preds[j].hits) ? preds[i].track_id : preds[j].track_id;
                to_delete.insert(victim_id);
            }
        }
    }

    // 削除実行
    for (int id : to_delete) {
        if (track_manager_->hasTrack(id)) {
            Track& t = track_manager_->getTrackMutable(id);
            t.track_state = TrackState::DELETED;
        }
    }
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

void MultiTargetTracker::setIMMNoiseMultipliers(float cv_mult, float bal_mult, float ct_mult, float sg_mult) {
    if (use_imm_) {
        if (imm_cpu_) {
            imm_cpu_->setModelNoiseMultipliers(cv_mult, bal_mult, ct_mult, sg_mult);
        }
        // GPU version would also need updating
    }
}

} // namespace fasttracker
