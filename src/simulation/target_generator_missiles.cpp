// ミサイル軌道生成の実装（target_generator.cppに統合予定）
#include "simulation/target_generator.hpp"
#include <cmath>

namespace fasttracker {

void TargetGenerator::generateBallisticMissileScenario(int num_missiles) {
    target_params_.clear();
    num_targets_ = num_missiles;

    for (int i = 0; i < num_missiles; i++) {
        TargetParameters params;
        params.motion_model = MotionModel::BALLISTIC_MISSILE;
        params.birth_time = i * 5.0;  // 5秒間隔で発射

        // 発射位置（レーダーから遠方）
        float launch_range = 50000.0f + uniform_dist_(rng_) * 50000.0f;  // 50-100km
        float launch_azimuth = (uniform_dist_(rng_) - 0.5f) * M_PI;

        params.initial_state(0) = launch_range * std::cos(launch_azimuth);
        params.initial_state(1) = launch_range * std::sin(launch_azimuth);
        params.initial_state(2) = 0.0f;
        params.initial_state(3) = 0.0f;

        // 弾道ミサイルパラメータ
        params.missile_params.launch_angle = 0.6f + uniform_dist_(rng_) * 0.4f;  // 35-60度
        params.missile_params.max_altitude = 30000.0f + uniform_dist_(rng_) * 40000.0f;  // 30-70km
        params.missile_params.boost_duration = 40.0f + uniform_dist_(rng_) * 40.0f;  // 40-80秒
        params.missile_params.boost_acceleration = 15.0f + uniform_dist_(rng_) * 10.0f;  // 1.5-2.5G

        // 着弾目標（レーダー付近）
        float impact_range = 5000.0f + uniform_dist_(rng_) * 10000.0f;
        float impact_azimuth = (uniform_dist_(rng_) - 0.5f) * 0.5f;
        params.missile_params.target_position(0) = impact_range * std::cos(impact_azimuth);
        params.missile_params.target_position(1) = impact_range * std::sin(impact_azimuth);

        // 飛翔時間を推定（簡易）
        params.death_time = params.birth_time + 180.0 + uniform_dist_(rng_) * 120.0;  // 3-5分

        target_params_.push_back(params);
    }
}

void TargetGenerator::generateHypersonicGlideScenario(int num_hgvs) {
    target_params_.clear();
    num_targets_ = num_hgvs;

    for (int i = 0; i < num_hgvs; i++) {
        TargetParameters params;
        params.motion_model = MotionModel::HYPERSONIC_GLIDE;
        params.birth_time = i * 10.0;  // 10秒間隔

        // 初期位置（高高度・遠方から進入）
        float initial_range = 100000.0f + uniform_dist_(rng_) * 50000.0f;  // 100-150km
        float initial_azimuth = (uniform_dist_(rng_) - 0.5f) * M_PI;

        params.initial_state(0) = initial_range * std::cos(initial_azimuth);
        params.initial_state(1) = initial_range * std::sin(initial_azimuth);

        // 初期速度（マッハ5-10）
        float initial_speed = 1500.0f + uniform_dist_(rng_) * 2000.0f;  // 1500-3500 m/s
        float velocity_azimuth = initial_azimuth + M_PI;  // レーダー方向へ

        params.initial_state(2) = initial_speed * std::cos(velocity_azimuth);
        params.initial_state(3) = initial_speed * std::sin(velocity_azimuth);

        // HGVパラメータ
        params.missile_params.cruise_altitude = 20000.0f + uniform_dist_(rng_) * 20000.0f;  // 20-40km
        params.missile_params.glide_ratio = 3.0f + uniform_dist_(rng_) * 3.0f;  // L/D = 3-6
        params.missile_params.skip_amplitude = 3000.0f + uniform_dist_(rng_) * 4000.0f;  // 3-7km
        params.missile_params.skip_frequency = 0.005f + uniform_dist_(rng_) * 0.015f;  // 0.005-0.02 Hz
        params.missile_params.terminal_dive_range = 10000.0f + uniform_dist_(rng_) * 15000.0f;  // 10-25km

        // 目標位置
        float target_range = 5000.0f + uniform_dist_(rng_) * 10000.0f;
        float target_azimuth = (uniform_dist_(rng_) - 0.5f) * 0.3f;
        params.missile_params.target_position(0) = target_range * std::cos(target_azimuth);
        params.missile_params.target_position(1) = target_range * std::sin(target_azimuth);

        params.death_time = params.birth_time + 60.0 + uniform_dist_(rng_) * 60.0;  // 1-2分

        target_params_.push_back(params);
    }
}

void TargetGenerator::generateMixedThreatScenario() {
    target_params_.clear();
    int num_aircraft = num_targets_ / 2;
    int num_ballistic = num_targets_ / 4;
    int num_hgv = num_targets_ - num_aircraft - num_ballistic;

    // 通常航空機
    initializeDefaultScenario();
    auto aircraft_params = target_params_;

    // 弾道ミサイル
    generateBallisticMissileScenario(num_ballistic);
    auto ballistic_params = target_params_;

    // HGV
    generateHypersonicGlideScenario(num_hgv);
    auto hgv_params = target_params_;

    // 統合
    target_params_.clear();
    target_params_.insert(target_params_.end(), aircraft_params.begin(), aircraft_params.end());
    target_params_.insert(target_params_.end(), ballistic_params.begin(), ballistic_params.end());
    target_params_.insert(target_params_.end(), hgv_params.begin(), hgv_params.end());

    num_targets_ = static_cast<int>(target_params_.size());
}

BallisticPhase TargetGenerator::getBallisticPhase(const TargetParameters& params, double time) const {
    double elapsed = time - params.birth_time;

    if (elapsed < params.missile_params.boost_duration) {
        return BallisticPhase::BOOST;
    }

    // 簡易的な判定: ブースト後の80%を中間段階、残り20%を終末段階とする
    double total_flight = params.death_time - params.birth_time;
    double midcourse_end = params.birth_time + total_flight * 0.8;

    if (time < midcourse_end) {
        return BallisticPhase::MIDCOURSE;
    } else {
        return BallisticPhase::TERMINAL;
    }
}

StateVector TargetGenerator::propagateBallisticMissile(const TargetParameters& params, double time) const {
    StateVector state = params.initial_state;
    double elapsed = time - params.birth_time;

    if (elapsed < 0) {
        state.setZero();
        return state;
    }

    BallisticPhase phase = getBallisticPhase(params, time);

    // 発射位置
    float x0 = params.initial_state(0);
    float y0 = params.initial_state(1);

    // 目標位置
    float xt = params.missile_params.target_position(0);
    float yt = params.missile_params.target_position(1);

    // 発射方向
    float dx = xt - x0;
    float dy = yt - y0;
    float range_to_target = std::sqrt(dx * dx + dy * dy);
    float azimuth = std::atan2(dy, dx);

    if (phase == BallisticPhase::BOOST) {
        // ブースト段階: 加速上昇
        float t = elapsed;
        float angle = params.missile_params.launch_angle;

        // 等加速度運動
        float ax = params.missile_params.boost_acceleration * std::cos(angle) * std::cos(azimuth);
        float ay = params.missile_params.boost_acceleration * std::cos(angle) * std::sin(azimuth);

        state(0) = x0 + 0.5f * ax * t * t;
        state(1) = y0 + 0.5f * ay * t * t;
        state(2) = ax * t;
        state(3) = ay * t;
        state(4) = ax;
        state(5) = ay;

    } else if (phase == BallisticPhase::MIDCOURSE) {
        // 中間段階: 放物線飛行（簡易モデル）
        float boost_end_time = params.missile_params.boost_duration;
        float t_midcourse = elapsed - boost_end_time;

        // ブースト終了時の速度
        float v_boost = params.missile_params.boost_acceleration * boost_end_time;
        float vx_boost = v_boost * std::cos(params.missile_params.launch_angle) * std::cos(azimuth);
        float vy_boost = v_boost * std::cos(params.missile_params.launch_angle) * std::sin(azimuth);

        // 自由落下（重力の影響は簡略化）
        state(0) = x0 + vx_boost * elapsed;
        state(1) = y0 + vy_boost * elapsed;
        state(2) = vx_boost;
        state(3) = vy_boost;
        state(4) = 0.0f;
        state(5) = 0.0f;

    } else {  // TERMINAL
        // 終末段階: 目標へ向けて機動
        float total_flight = params.death_time - params.birth_time;
        float terminal_start = total_flight * 0.8;
        float t_terminal = elapsed - terminal_start;
        float terminal_duration = total_flight - terminal_start;

        // 目標へ向かう直線軌道（簡易）
        float progress = t_terminal / terminal_duration;

        // 現在位置から目標への補間
        float x_terminal_start = x0 + (xt - x0) * 0.8f;
        float y_terminal_start = y0 + (yt - y0) * 0.8f;

        state(0) = x_terminal_start + (xt - x_terminal_start) * progress;
        state(1) = y_terminal_start + (yt - y_terminal_start) * progress;

        // 終末速度（高速）
        float terminal_speed = 1000.0f + uniform_dist_(rng_) * 500.0f;
        state(2) = terminal_speed * std::cos(azimuth);
        state(3) = terminal_speed * std::sin(azimuth);

        // 終末機動（簡易的なweave）
        float maneuver_accel = 50.0f;  // 5G
        state(4) = maneuver_accel * std::sin(t_terminal * 2.0);
        state(5) = maneuver_accel * std::cos(t_terminal * 2.0);
    }

    return state;
}

StateVector TargetGenerator::propagateHypersonicGlide(const TargetParameters& params, double time) const {
    StateVector state = params.initial_state;
    double elapsed = time - params.birth_time;

    if (elapsed < 0) {
        state.setZero();
        return state;
    }

    // 初期位置・速度
    float x0 = params.initial_state(0);
    float y0 = params.initial_state(1);
    float vx0 = params.initial_state(2);
    float vy0 = params.initial_state(3);

    // 目標位置
    float xt = params.missile_params.target_position(0);
    float yt = params.missile_params.target_position(1);

    // 目標への方向
    float dx = xt - x0;
    float dy = yt - y0;
    float range_to_target = std::sqrt(dx * dx + dy * dy);
    float azimuth_to_target = std::atan2(dy, dx);

    // 巡航速度（マッハ5-10程度）
    float cruise_speed = std::sqrt(vx0 * vx0 + vy0 * vy0);

    // 現在の距離
    float current_range = range_to_target * (1.0f - elapsed / (params.death_time - params.birth_time));

    if (current_range > params.missile_params.terminal_dive_range) {
        // 巡航段階: スキップグライド
        float t = elapsed;

        // 基本的な直線飛行
        float progress = elapsed / (params.death_time - params.birth_time);
        float base_x = x0 + (xt - x0) * progress;
        float base_y = y0 + (yt - y0) * progress;

        // スキップ機動（周期的な上下運動）
        float skip_phase = 2.0f * M_PI * params.missile_params.skip_frequency * t;
        float skip_offset_x = params.missile_params.skip_amplitude * std::sin(skip_phase) *
                             std::cos(azimuth_to_target + M_PI / 2.0f);
        float skip_offset_y = params.missile_params.skip_amplitude * std::sin(skip_phase) *
                             std::sin(azimuth_to_target + M_PI / 2.0f);

        state(0) = base_x + skip_offset_x;
        state(1) = base_y + skip_offset_y;

        // 速度（巡航速度を維持）
        float vx_cruise = cruise_speed * std::cos(azimuth_to_target);
        float vy_cruise = cruise_speed * std::sin(azimuth_to_target);

        // スキップによる速度変化
        float skip_vel_scale = std::cos(skip_phase);
        state(2) = vx_cruise * (1.0f + 0.2f * skip_vel_scale);
        state(3) = vy_cruise * (1.0f + 0.2f * skip_vel_scale);

        // 機動加速度
        state(4) = -50.0f * std::sin(skip_phase) * std::cos(azimuth_to_target + M_PI / 2.0f);
        state(5) = -50.0f * std::sin(skip_phase) * std::sin(azimuth_to_target + M_PI / 2.0f);

    } else {
        // 終末ダイブ段階
        float terminal_progress = (params.missile_params.terminal_dive_range - current_range) /
                                 params.missile_params.terminal_dive_range;

        // 加速しながら目標へ
        float dive_speed = cruise_speed * (1.0f + terminal_progress * 0.5f);  // 50%加速

        state(0) = xt - (xt - x0) * (1.0f - elapsed / (params.death_time - params.birth_time));
        state(1) = yt - (yt - y0) * (1.0f - elapsed / (params.death_time - params.birth_time));
        state(2) = dive_speed * std::cos(azimuth_to_target);
        state(3) = dive_speed * std::sin(azimuth_to_target);

        // 終末機動（unpredictable）
        float maneuver_freq = 3.0f;
        state(4) = 80.0f * std::sin(maneuver_freq * elapsed);  // 8G機動
        state(5) = 80.0f * std::cos(maneuver_freq * elapsed * 1.3f);
    }

    return state;
}

} // namespace fasttracker
