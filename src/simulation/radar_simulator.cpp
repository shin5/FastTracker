#include "simulation/radar_simulator.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

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

    // 真の目標状態を取得
    std::vector<StateVector> true_states = target_gen_.generateStates(time);
    std::vector<int> active_ids = target_gen_.getActiveTargets(time);
    total_targets_ += static_cast<int>(true_states.size());

    // 各目標からの観測を生成
    for (size_t i = 0; i < true_states.size(); i++) {
        const auto& state = true_states[i];

        // 視野内チェック
        if (!isInFieldOfView(state)) continue;

        // ビームステアリング: アクティブビーム内かチェック
        {
            float dx = state(0) - params_.sensor_x;
            float dy = state(1) - params_.sensor_y;
            float az = std::atan2(dy, dx);
            if (!isOnBeam(az)) continue;
        }

        // 観測生成（Swerling II SNRサンプリングを含む）
        Measurement meas = stateToMeasurement(state);

        // Swerling II 検出判定: CFAR閾値 γ = −ln(Pfa) と瞬時SNRを比較
        if (!isDetectedSwerlingII(meas.snr)) {
            total_missed_++;
            continue;
        }

        addNoise(meas);
        meas.timestamp = time;
        meas.sensor_id = sensor_id_;

        measurements.push_back(meas);
        int target_id = (i < active_ids.size()) ? active_ids[i] : static_cast<int>(i);
        true_associations_.push_back(target_id);
        total_detections_++;
    }

    // クラッタ生成
    auto clutter = generateClutter(time);
    for (auto& c : clutter) {
        c.is_clutter = true;
        measurements.push_back(c);
        true_associations_.push_back(-1);  // -1 = クラッタ
    }
    total_clutter_ += static_cast<int>(clutter.size());

    // シャッフル（観測順序をランダム化）
    std::vector<int> indices(measurements.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = static_cast<int>(i);
    std::shuffle(indices.begin(), indices.end(), rng_);

    std::vector<Measurement> shuffled_meas;
    std::vector<int> shuffled_assoc;
    for (int idx : indices) {
        shuffled_meas.push_back(measurements[idx]);
        shuffled_assoc.push_back(true_associations_[idx]);
    }

    measurements = shuffled_meas;
    true_associations_ = shuffled_assoc;

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

    // 監視領域の面積
    float surveillance_area;
    if (!beam_directions_.empty()) {
        // ビームステアリング時: ビームセクタの合計面積
        surveillance_area = static_cast<float>(beam_directions_.size()) *
                           0.5f * params_.beam_width * params_.max_range * params_.max_range;
    } else {
        surveillance_area = M_PI * params_.max_range * params_.max_range;
    }

    // クラッタ数の期待値
    float lambda = params_.false_alarm_rate * surveillance_area;

    // ポアソン分布でクラッタ数を決定
    poisson_dist_ = std::poisson_distribution<int>(lambda);
    int num_clutter = poisson_dist_(rng_);

    for (int i = 0; i < num_clutter; i++) {
        Measurement meas;

        // ランダムな位置にクラッタを生成
        float r = params_.max_range * std::sqrt(uniform_dist_(rng_));
        float theta;

        if (!beam_directions_.empty()) {
            // ビーム内にクラッタを配置
            int beam_idx = static_cast<int>(uniform_dist_(rng_) * beam_directions_.size());
            if (beam_idx >= static_cast<int>(beam_directions_.size()))
                beam_idx = static_cast<int>(beam_directions_.size()) - 1;
            float offset = (uniform_dist_(rng_) - 0.5f) * params_.beam_width;
            theta = beam_directions_[beam_idx] + offset;
            // [-π, π] に正規化
            while (theta > static_cast<float>(M_PI)) theta -= 2.0f * static_cast<float>(M_PI);
            while (theta < -static_cast<float>(M_PI)) theta += 2.0f * static_cast<float>(M_PI);
        } else {
            theta = (uniform_dist_(rng_) - 0.5f) * params_.field_of_view;
        }

        meas.range = r;
        meas.azimuth = theta;
        meas.elevation = 0.0f;
        meas.doppler = (uniform_dist_(rng_) - 0.5f) * 200.0f;  // -100~100 m/s

        // クラッタは低SN比（0-15 dB程度）
        meas.snr = uniform_dist_(rng_) * 15.0f;

        // ノイズ付加
        addNoise(meas);

        meas.timestamp = time;
        meas.sensor_id = sensor_id_;

        clutter.push_back(meas);
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
    if (range > params_.max_range) return false;

    float azimuth = std::atan2(dy, dx);
    float half_fov = params_.field_of_view / 2.0f;

    // アンテナ中心方位からの角度差
    float diff = azimuth - params_.antenna_boresight;
    while (diff > static_cast<float>(M_PI)) diff -= 2.0f * static_cast<float>(M_PI);
    while (diff < -static_cast<float>(M_PI)) diff += 2.0f * static_cast<float>(M_PI);

    return (std::fabs(diff) <= half_fov);
}

bool RadarSimulator::isOnBeam(float azimuth) const {
    if (beam_directions_.empty()) return false;

    float half_bw = params_.beam_width / 2.0f;
    for (float beam_center : beam_directions_) {
        float diff = azimuth - beam_center;
        while (diff > static_cast<float>(M_PI)) diff -= 2.0f * static_cast<float>(M_PI);
        while (diff < -static_cast<float>(M_PI)) diff += 2.0f * static_cast<float>(M_PI);
        if (std::fabs(diff) <= half_bw) return true;
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
