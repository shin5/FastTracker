#include "simulation/radar_simulator.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace fasttracker {

RadarSimulator::RadarSimulator(const TargetGenerator& target_gen,
                               const RadarParameters& params,
                               int sensor_id)
    : target_gen_(target_gen),
      params_(params),
      sensor_id_(sensor_id),
      rng_(std::random_device{}()),
      uniform_dist_(0.0f, 1.0f),
      normal_dist_(0.0f, 1.0f),
      poisson_dist_(10),
      total_targets_(0),
      total_detections_(0),
      total_clutter_(0),
      total_missed_(0)
{
    std::cout << "RadarSimulator initialized" << std::endl;
    std::cout << "  Detection probability: " << params_.detection_probability << std::endl;
    std::cout << "  False alarm rate: " << params_.false_alarm_rate << std::endl;
    std::cout << "  Max range: " << params_.max_range << " m" << std::endl;
}

std::vector<Measurement> RadarSimulator::generate(double time) {
    std::vector<Measurement> measurements;
    true_associations_.clear();
    beam_types_.clear();

    // 真の目標状態を取得
    std::vector<StateVector> true_states = target_gen_.generateStates(time);
    std::vector<int> active_ids = target_gen_.getActiveTargets(time);
    total_targets_ += static_cast<int>(true_states.size());

    // === Phase 1: 各目標の探知判定（ノイズ付加前） ===
    struct PreDetection {
        Measurement meas;   // ノイズ付加前の観測
        int beam_idx;       // 探知ビームインデックス
        int target_id;      // 真の目標ID
        float snr_linear;   // 線形SNR（電力加算・重み付け用）
    };
    std::vector<PreDetection> pre_detections;

    for (size_t i = 0; i < true_states.size(); i++) {
        const auto& state = true_states[i];

        // レーダー覆域内チェック（距離＋仰角のみ、方位角制限なし）
        if (!isInRadarCoverage(state)) continue;

        // ビームステアリング: アクティブビーム内かチェック（方位角・仰角の両方で判定）
        int det_beam_idx = -1;
        {
            float dx = state(0) - params_.sensor_x;
            float dy = state(1) - params_.sensor_y;
            float dz = state(2) - params_.sensor_z;
            float r_horiz = std::sqrt(dx * dx + dy * dy);
            float az = std::atan2(dy, dx);
            float el = std::atan2(dz, r_horiz);
            if (!isOnBeamWithIndex(az, el, det_beam_idx)) {
                // 診断ログ: ビームミスの詳細（最初の10回のみ）
                static int miss_log_count = 0;
                if (miss_log_count < 10 && !beam_directions_.empty()) {
                    float half_bw = params_.beam_width / 2.0f;
                    std::cout << "[Beam Miss] Target el=" << (el * 180.0f / M_PI)
                              << "deg, az=" << (az * 180.0f / M_PI) << "deg | Beams: ";
                    for (size_t bi = 0; bi < std::min(beam_directions_.size(), size_t(3)); bi++) {
                        float beam_el = (bi < beam_elevations_.size())
                                        ? beam_elevations_[bi]
                                        : 0.0f;
                        float diff_el = std::fabs(el - beam_el) * 180.0f / M_PI;
                        float diff_az = std::fabs(az - beam_directions_[bi]) * 180.0f / M_PI;
                        std::cout << "[" << bi << "] el=" << (beam_el * 180.0f / M_PI)
                                  << "deg (Δ=" << diff_el << "), az="
                                  << (beam_directions_[bi] * 180.0f / M_PI)
                                  << "deg (Δ=" << diff_az << ") ";
                    }
                    std::cout << "| half_bw=" << (half_bw * 180.0f / M_PI) << "deg" << std::endl;
                    miss_log_count++;
                }
                continue;
            }
        }

        // 観測生成（Swerling II SNRサンプリングを含む）
        Measurement meas = stateToMeasurement(state);

        // Swerling II 検出判定: CFAR閾値 γ = −ln(Pfa) と瞬時SNRを比較
        if (!isDetectedSwerlingII(meas.snr)) {
            total_missed_++;
            continue;
        }

        int target_id = (i < active_ids.size()) ? active_ids[i] : static_cast<int>(i);
        float snr_lin = std::pow(10.0f, meas.snr / 10.0f);
        pre_detections.push_back({meas, det_beam_idx, target_id, snr_lin});
        total_detections_++;
    }

    // === Phase 2: レンジ分解能セル内の未分解目標をマージ ===
    // 同一ビーム内で距離差 < range_resolution の目標群を1つの合成観測に統合
    float res = params_.range_resolution;
    if (res > 0.0f && pre_detections.size() > 1) {
        // ビームインデックスでグループ化
        std::unordered_map<int, std::vector<size_t>> beam_groups;
        for (size_t i = 0; i < pre_detections.size(); i++) {
            beam_groups[pre_detections[i].beam_idx].push_back(i);
        }

        std::vector<PreDetection> merged_detections;
        static int merge_log_count = 0;

        for (auto& [bidx, indices] : beam_groups) {
            if (indices.size() <= 1) {
                for (size_t idx : indices) {
                    merged_detections.push_back(pre_detections[idx]);
                }
                continue;
            }

            // レンジ順にソート
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return pre_detections[a].meas.range < pre_detections[b].meas.range;
            });

            // 固定窓クラスタリング: クラスタ先頭からの距離差 < range_resolution でグループ化
            // チェーン効果を防止（A-B未分解、B-C未分解でもA-C分解なら別クラスタ）
            std::vector<std::vector<size_t>> clusters;
            clusters.push_back({indices[0]});
            float cluster_anchor = pre_detections[indices[0]].meas.range;
            for (size_t j = 1; j < indices.size(); j++) {
                float curr_range = pre_detections[indices[j]].meas.range;
                if (curr_range - cluster_anchor < res) {
                    clusters.back().push_back(indices[j]);
                } else {
                    cluster_anchor = curr_range;
                    clusters.push_back({indices[j]});
                }
            }

            for (auto& cluster : clusters) {
                if (cluster.size() == 1) {
                    merged_detections.push_back(pre_detections[cluster[0]]);
                } else {
                    // 電力加重平均で合成観測を生成
                    float total_snr_lin = 0.0f;
                    float w_range = 0.0f, w_az = 0.0f, w_el = 0.0f, w_doppler = 0.0f;
                    int dominant_id = -1;
                    float max_snr = -1.0f;

                    for (size_t idx : cluster) {
                        auto& pd = pre_detections[idx];
                        float w = pd.snr_linear;
                        total_snr_lin += w;
                        w_range += w * pd.meas.range;
                        w_az += w * pd.meas.azimuth;
                        w_el += w * pd.meas.elevation;
                        w_doppler += w * pd.meas.doppler;
                        if (w > max_snr) {
                            max_snr = w;
                            dominant_id = pd.target_id;
                        }
                    }

                    PreDetection merged;
                    merged.beam_idx = pre_detections[cluster[0]].beam_idx;
                    merged.target_id = dominant_id;  // 最大SNR目標のIDを継承
                    merged.snr_linear = total_snr_lin;
                    merged.meas.range = w_range / total_snr_lin;
                    merged.meas.azimuth = w_az / total_snr_lin;
                    merged.meas.elevation = w_el / total_snr_lin;
                    merged.meas.doppler = w_doppler / total_snr_lin;
                    merged.meas.snr = 10.0f * std::log10(total_snr_lin);  // 非干渉電力加算

                    if (merge_log_count < 20) {
                        float range_spread = pre_detections[cluster.back()].meas.range
                                           - pre_detections[cluster[0]].meas.range;
                        std::cout << "[Resolution Merge] " << cluster.size()
                                  << " targets in beam " << merged.beam_idx
                                  << " merged (Δr=" << range_spread << "m < "
                                  << res << "m), dominant=tgt" << dominant_id
                                  << ", SNR=" << merged.meas.snr << "dB" << std::endl;
                        merge_log_count++;
                    }

                    merged_detections.push_back(merged);
                }
            }
        }
        pre_detections = merged_detections;
    }

    // === Phase 3: ノイズ付加・出力構築 ===
    for (auto& pd : pre_detections) {
        addNoise(pd.meas);
        pd.meas.timestamp = time;
        pd.meas.sensor_id = sensor_id_;

        measurements.push_back(pd.meas);
        true_associations_.push_back(pd.target_id);
        // ビーム種別: beam_target_ranges_[beam_idx] > 0 なら追尾ビーム
        int btype = 0;  // デフォルト: サーチ
        if (pd.beam_idx >= 0 && pd.beam_idx < static_cast<int>(beam_target_ranges_.size())
            && beam_target_ranges_[pd.beam_idx] > 0.0f) {
            btype = 1;  // 追尾ビーム
        }
        beam_types_.push_back(btype);
    }

    // クラッタ生成
    auto clutter = generateClutter(time);
    for (size_t ci = 0; ci < clutter.size(); ci++) {
        measurements.push_back(clutter[ci]);
        true_associations_.push_back(-1);  // -1 = クラッタ（評価用のみ）
        int cbtype = (ci < clutter_beam_types_.size()) ? clutter_beam_types_[ci] : 0;
        beam_types_.push_back(cbtype);
    }
    total_clutter_ += static_cast<int>(clutter.size());

    // シャッフル（観測順序をランダム化）
    std::vector<int> indices(measurements.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = static_cast<int>(i);
    std::shuffle(indices.begin(), indices.end(), rng_);

    std::vector<Measurement> shuffled_meas;
    std::vector<int> shuffled_assoc;
    std::vector<int> shuffled_beam_types;
    for (int idx : indices) {
        shuffled_meas.push_back(measurements[idx]);
        shuffled_assoc.push_back(true_associations_[idx]);
        shuffled_beam_types.push_back(beam_types_[idx]);
    }

    measurements = shuffled_meas;
    true_associations_ = shuffled_assoc;
    beam_types_ = shuffled_beam_types;

    return measurements;
}

std::vector<StateVector> RadarSimulator::getTrueStates(double time) const {
    return target_gen_.generateStates(time);
}

Measurement RadarSimulator::stateToMeasurement(const StateVector& state) const {
    Measurement meas;

    // センサーからの相対座標（3D）
    float dx = state(0) - params_.sensor_x;
    float dy = state(1) - params_.sensor_y;
    float dz = state(2) - params_.sensor_z;
    float vx = state(3);
    float vy = state(4);
    float vz = state(5);

    // Range（センサーからの3D距離）
    meas.range = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Azimuth（センサーから見た方位）
    meas.azimuth = std::atan2(dy, dx);

    // Elevation（仰角）
    float r_horiz = std::sqrt(dx * dx + dy * dy);
    meas.elevation = std::atan2(dz, r_horiz);

    // Doppler（視線方向の速度成分, 3D）
    if (meas.range > 1e-6f) {
        meas.doppler = (dx * vx + dy * vy + dz * vz) / meas.range;
    } else {
        meas.doppler = 0.0f;
    }

    // SN比の計算（簡易レーダー方程式）: SNR_avg = SNR_ref + 40*log10(R_ref/R)
    float snr_ref = params_.snr_ref;  // 基準SN比 [dB] @ 1km
    float r_ref = 1000.0f;             // 基準距離 [m]
    float range_ratio = r_ref / std::max(meas.range, 100.0f);
    float snr_avg_dB = snr_ref + 40.0f * std::log10(range_ratio);  // R^4則（平均SNR）

    // Swerling II モデル: パルスごとに独立なRCS変動
    // RCS ~ Exp(σ_avg)  →  SNR_inst = SNR_avg * X, X ~ Exp(1) = -ln(U)
    float snr_avg_lin = std::pow(10.0f, snr_avg_dB / 10.0f);
    float u = std::max(uniform_dist_(rng_), 1e-10f);
    float rcs_factor = -std::log(u);          // Exp(1) サンプル
    float snr_inst_lin = snr_avg_lin * rcs_factor;
    meas.snr = 10.0f * std::log10(std::max(snr_inst_lin, 1e-10f));  // 瞬時SNR [dB]

    return meas;
}

void RadarSimulator::addNoise(Measurement& meas) {
    meas.range += normal_dist_(rng_) * params_.meas_noise.range_noise;
    meas.azimuth += normal_dist_(rng_) * params_.meas_noise.azimuth_noise;
    meas.elevation += normal_dist_(rng_) * params_.meas_noise.elevation_noise;
    meas.doppler += normal_dist_(rng_) * params_.meas_noise.doppler_noise;

    // SN比にノイズを付加（±2dB程度のばらつき）
    meas.snr += normal_dist_(rng_) * 2.0f;

    // Range は正の値に制限
    if (meas.range < 0.0f) meas.range = 0.0f;

    // Azimuth を -π ~ π に正規化
    while (meas.azimuth > M_PI) meas.azimuth -= 2.0f * M_PI;
    while (meas.azimuth < -M_PI) meas.azimuth += 2.0f * M_PI;
}

std::vector<Measurement> RadarSimulator::generateClutter(double time) {
    std::vector<Measurement> clutter;
    clutter_beam_types_.clear();

    // サーチ領域の有効範囲を決定
    float search_min = params_.search_min_range;
    float search_max = (params_.search_max_range > 0.0f && params_.search_max_range < params_.max_range)
                       ? params_.search_max_range
                       : params_.max_range;

    // ビーム種別ごとのインデックスを分類
    std::vector<int> search_beam_indices;  // サーチビーム (beam_target_ranges == 0)
    std::vector<int> track_beam_indices;   // 追尾ビーム (beam_target_ranges > 0)

    // 監視領域の面積を計算
    float search_surveillance_area = 0.0f;
    float track_surveillance_area = 0.0f;

    if (!beam_directions_.empty()) {
        for (size_t b = 0; b < beam_directions_.size(); b++) {
            float btr = (b < beam_target_ranges_.size()) ? beam_target_ranges_[b] : 0.0f;
            if (btr > 0.0f) {
                // 追尾ビーム: 目標距離±track_range_width/2 の扇環形
                track_beam_indices.push_back(static_cast<int>(b));
                float half_w = params_.track_range_width / 2.0f;
                if (half_w > 0.0f) {
                    float r_min_t = std::max(0.0f, btr - half_w);
                    float r_max_t = btr + half_w;
                    track_surveillance_area += 0.5f * params_.beam_width * (r_max_t * r_max_t - r_min_t * r_min_t);
                }
            } else if (btr == 0.0f) {
                // サーチビーム: search_min ~ search_max の扇環形
                search_beam_indices.push_back(static_cast<int>(b));
                search_surveillance_area += 0.5f * params_.beam_width * (search_max * search_max - search_min * search_min);
            }
        }
    } else {
        search_surveillance_area = M_PI * (search_max * search_max - search_min * search_min);
    }

    // --- サーチビームのクラッタ ---
    float search_lambda = params_.false_alarm_rate * search_surveillance_area;
    int num_search_clutter = 0;
    if (search_lambda > 0.0f) {
        poisson_dist_ = std::poisson_distribution<int>(search_lambda);
        num_search_clutter = poisson_dist_(rng_);
    }

    for (int i = 0; i < num_search_clutter; i++) {
        Measurement meas;
        float theta, clutter_el, r;

        if (!beam_directions_.empty() && !search_beam_indices.empty()) {
            int si = static_cast<int>(uniform_dist_(rng_) * search_beam_indices.size());
            if (si >= static_cast<int>(search_beam_indices.size()))
                si = static_cast<int>(search_beam_indices.size()) - 1;
            int beam_idx = search_beam_indices[si];

            float u = uniform_dist_(rng_);
            float r_sq = search_min * search_min + (search_max * search_max - search_min * search_min) * u;
            r = std::sqrt(r_sq);

            theta = beam_directions_[beam_idx] + (uniform_dist_(rng_) - 0.5f) * params_.beam_width;
            while (theta > static_cast<float>(M_PI)) theta -= 2.0f * static_cast<float>(M_PI);
            while (theta < -static_cast<float>(M_PI)) theta += 2.0f * static_cast<float>(M_PI);

            float beam_el = (beam_idx < static_cast<int>(beam_elevations_.size()))
                            ? beam_elevations_[beam_idx] : 0.0f;
            clutter_el = beam_el + (uniform_dist_(rng_) - 0.5f) * params_.beam_width;
        } else {
            float u = uniform_dist_(rng_);
            float r_sq = search_min * search_min + (search_max * search_max - search_min * search_min) * u;
            r = std::sqrt(r_sq);
            theta = params_.antenna_boresight + (uniform_dist_(rng_) - 0.5f) * params_.azimuth_coverage;
            while (theta > static_cast<float>(M_PI)) theta -= 2.0f * static_cast<float>(M_PI);
            while (theta < -static_cast<float>(M_PI)) theta += 2.0f * static_cast<float>(M_PI);
            clutter_el = (uniform_dist_(rng_) - 0.5f) * params_.beam_width;
        }

        meas.range = r;
        meas.azimuth = theta;
        meas.elevation = clutter_el;
        meas.doppler = (uniform_dist_(rng_) - 0.5f) * 200.0f;
        meas.snr = uniform_dist_(rng_) * 15.0f;
        addNoise(meas);
        meas.timestamp = time;
        meas.sensor_id = sensor_id_;
        clutter.push_back(meas);
        clutter_beam_types_.push_back(0);  // サーチビーム
    }

    // --- 追尾ビームのクラッタ ---
    float track_lambda = params_.false_alarm_rate * track_surveillance_area;
    int num_track_clutter = 0;
    if (track_lambda > 0.0f) {
        poisson_dist_ = std::poisson_distribution<int>(track_lambda);
        num_track_clutter = poisson_dist_(rng_);
    }

    for (int i = 0; i < num_track_clutter; i++) {
        Measurement meas;

        // 追尾ビームをランダム選択（面積比重み付き）
        // 簡易: 均一選択（各追尾ビームの距離幅は同じ track_range_width）
        int ti = static_cast<int>(uniform_dist_(rng_) * track_beam_indices.size());
        if (ti >= static_cast<int>(track_beam_indices.size()))
            ti = static_cast<int>(track_beam_indices.size()) - 1;
        int beam_idx = track_beam_indices[ti];

        float target_range = beam_target_ranges_[beam_idx];
        float half_w = params_.track_range_width / 2.0f;
        float r_min_t = std::max(0.0f, target_range - half_w);
        float r_max_t = target_range + half_w;

        // 扇環形内で均一分布（r² 分布）
        float u = uniform_dist_(rng_);
        float r_sq = r_min_t * r_min_t + (r_max_t * r_max_t - r_min_t * r_min_t) * u;
        float r = std::sqrt(r_sq);

        float theta = beam_directions_[beam_idx] + (uniform_dist_(rng_) - 0.5f) * params_.beam_width;
        while (theta > static_cast<float>(M_PI)) theta -= 2.0f * static_cast<float>(M_PI);
        while (theta < -static_cast<float>(M_PI)) theta += 2.0f * static_cast<float>(M_PI);

        float beam_el = (beam_idx < static_cast<int>(beam_elevations_.size()))
                        ? beam_elevations_[beam_idx] : 0.0f;
        float clutter_el = beam_el + (uniform_dist_(rng_) - 0.5f) * params_.beam_width;

        meas.range = r;
        meas.azimuth = theta;
        meas.elevation = clutter_el;
        meas.doppler = (uniform_dist_(rng_) - 0.5f) * 200.0f;
        meas.snr = uniform_dist_(rng_) * 15.0f;
        addNoise(meas);
        meas.timestamp = time;
        meas.sensor_id = sensor_id_;
        clutter.push_back(meas);
        clutter_beam_types_.push_back(1);  // 追尾ビーム
    }

    return clutter;
}

bool RadarSimulator::isDetected(const StateVector& state) const {
    float detection_prob = params_.detection_probability;

    // センサーからの3D相対距離
    float dx = state(0) - params_.sensor_x;
    float dy = state(1) - params_.sensor_y;
    float dz = state(2) - params_.sensor_z;
    float range = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (range > params_.max_range) {
        return false;
    }

    // レーダー方程式の簡易版（距離の4乗に反比例）
    float range_factor = std::pow(params_.max_range / std::max(range, 100.0f), 4.0f);
    detection_prob = std::min(detection_prob * range_factor, 1.0f);

    return uniform_dist_(rng_) < detection_prob;
}

bool RadarSimulator::isDetectedSwerlingII(float snr_inst_dB) const {
    // Neyman-Pearson CFAR閾値: γ = −ln(Pfa)
    // 瞬時SNR（線形）> γ ならば検出
    float gamma_T = -std::log(std::max(params_.pfa_per_pulse, 1e-30f));
    float snr_inst_lin = std::pow(10.0f, snr_inst_dB / 10.0f);
    return snr_inst_lin > gamma_T;
}

bool RadarSimulator::isInFieldOfView(const StateVector& state) const {
    // センサーからの3D相対座標
    float dx = state(0) - params_.sensor_x;
    float dy = state(1) - params_.sensor_y;
    float dz = state(2) - params_.sensor_z;

    float range = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (range < params_.min_range || range > params_.max_range) return false;

    float azimuth = std::atan2(dy, dx);
    float half_fov = params_.azimuth_coverage / 2.0f;

    // アンテナ中心方位からの角度差
    float diff = azimuth - params_.antenna_boresight;
    while (diff > static_cast<float>(M_PI)) diff -= 2.0f * static_cast<float>(M_PI);
    while (diff < -static_cast<float>(M_PI)) diff += 2.0f * static_cast<float>(M_PI);

    return (std::fabs(diff) <= half_fov);
}

bool RadarSimulator::isInRadarCoverage(const StateVector& state) const {
    // センサーからの3D相対座標
    float dx = state(0) - params_.sensor_x;
    float dy = state(1) - params_.sensor_y;
    float dz = state(2) - params_.sensor_z;

    // Range check
    float range = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (range < params_.min_range || range > params_.max_range) return false;

    // Elevation check
    float r_horiz = std::sqrt(dx * dx + dy * dy);
    float elevation = std::atan2(dz, r_horiz);
    if (elevation < params_.min_elevation || elevation > params_.max_elevation)
        return false;

    // NO azimuth check - track beams can illuminate anywhere within physical limits
    return true;
}

bool RadarSimulator::isOnBeam(float azimuth, float elevation) const {
    int dummy;
    return isOnBeamWithIndex(azimuth, elevation, dummy);
}

bool RadarSimulator::isOnBeamWithIndex(float azimuth, float elevation, int& beam_idx) const {
    beam_idx = -1;
    // ビーム方向が未設定の場合は、ビームステアリングなし（FOV全体を検出）
    if (beam_directions_.empty()) return true;

    float half_bw = params_.beam_width / 2.0f;
    for (size_t i = 0; i < beam_directions_.size(); i++) {
        // 仰角: beam_elevations_ が設定されていれば対応する値を使用、
        // 未設定の場合は仰角0°にフォールバック
        float beam_el = (i < beam_elevations_.size())
                        ? beam_elevations_[i]
                        : 0.0f;

        // 方位角チェック
        float diff_az = azimuth - beam_directions_[i];
        while (diff_az >  static_cast<float>(M_PI)) diff_az -= 2.0f * static_cast<float>(M_PI);
        while (diff_az < -static_cast<float>(M_PI)) diff_az += 2.0f * static_cast<float>(M_PI);
        if (std::fabs(diff_az) > half_bw) continue;

        // 仰角チェック（方位角が通過した場合のみ評価）
        float diff_el = elevation - beam_el;
        if (std::fabs(diff_el) <= half_bw) {
            beam_idx = static_cast<int>(i);
            return true;
        }
    }
    return false;
}

void RadarSimulator::resetStatistics() {
    total_targets_ = 0;
    total_detections_ = 0;
    total_clutter_ = 0;
    total_missed_ = 0;
}

void RadarSimulator::printStatistics() const {
    std::cout << "=== Radar Simulator Statistics ===" << std::endl;
    std::cout << "Total targets: " << total_targets_ << std::endl;
    std::cout << "Detections: " << total_detections_ << std::endl;
    std::cout << "Missed: " << total_missed_ << std::endl;
    std::cout << "Clutter: " << total_clutter_ << std::endl;

    if (total_targets_ > 0) {
        float detection_rate = static_cast<float>(total_detections_) / total_targets_;
        std::cout << "Detection rate: " << (detection_rate * 100.0f) << "%" << std::endl;
    }

    if (total_detections_ + total_clutter_ > 0) {
        float clutter_ratio = static_cast<float>(total_clutter_) /
                             (total_detections_ + total_clutter_);
        std::cout << "Clutter ratio: " << (clutter_ratio * 100.0f) << "%" << std::endl;
    }

    std::cout << "===================================" << std::endl;
}

} // namespace fasttracker
