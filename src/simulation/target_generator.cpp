#include "simulation/target_generator.hpp"
#include <cmath>
#include <iostream>

namespace fasttracker {

TargetGenerator::TargetGenerator(int num_targets, float area_size, float max_velocity)
    : num_targets_(num_targets),
      area_size_(area_size),
      max_velocity_(max_velocity),
      rng_(std::random_device{}()),
      uniform_dist_(0.0f, 1.0f),
      normal_dist_(0.0f, 1.0f)
{
    target_params_.resize(num_targets_);
}

void TargetGenerator::initializeDefaultScenario() {
    std::cout << "Initializing default scenario with " << num_targets_ << " targets..." << std::endl;

    for (int i = 0; i < num_targets_; i++) {
        auto& param = target_params_[i];
        param.initial_state = generateRandomInitialState();
        param.birth_time = 0.0;
        param.death_time = -1.0;  // 無限

        // モーションモデルの割り当て
        float rand = uniform_dist_(rng_);
        if (rand < 0.7f) {
            // 70%: 等速直線運動
            param.motion_model = MotionModel::CONSTANT_VELOCITY;
        } else if (rand < 0.9f) {
            // 20%: 等加速度運動
            param.motion_model = MotionModel::CONSTANT_ACCELERATION;
            float accel_dir = uniform_dist_(rng_) * 2.0f * M_PI;
            param.initial_state(6) = 2.0f * std::cos(accel_dir);  // ax
            param.initial_state(7) = 2.0f * std::sin(accel_dir);  // ay
        } else {
            // 10%: 機動目標
            param.motion_model = MotionModel::MANEUVERING;
            param.turn_rate = (uniform_dist_(rng_) - 0.5f) * 0.1f;  // -0.05~0.05 rad/s
            param.accel_magnitude = uniform_dist_(rng_) * 5.0f;     // 0~5 m/s²
        }
    }

    std::cout << "  70% constant velocity, 20% constant acceleration, 10% maneuvering" << std::endl;
}

void TargetGenerator::initializeCustomScenario(const std::vector<TargetParameters>& target_params) {
    num_targets_ = static_cast<int>(target_params.size());
    target_params_ = target_params;
}

StateVector TargetGenerator::generateRandomInitialState() {
    StateVector state;

    // 位置: 領域内にランダム配置
    float x = (uniform_dist_(rng_) - 0.5f) * area_size_;
    float y = (uniform_dist_(rng_) - 0.5f) * area_size_;

    // 速度: ランダムな方向と大きさ
    float speed = uniform_dist_(rng_) * max_velocity_;
    float heading = uniform_dist_(rng_) * 2.0f * M_PI;
    float vx = speed * std::cos(heading);
    float vy = speed * std::sin(heading);

    // 加速度: 初期は0
    float ax = 0.0f;
    float ay = 0.0f;

    state << x, y, 0.0f, vx, vy, 0.0f, ax, ay, 0.0f;
    return state;
}

std::vector<StateVector> TargetGenerator::generateStates(double time) const {
    std::vector<StateVector> states;
    states.reserve(num_targets_);

    // 高度キャッシュをリサイズ
    if (last_altitudes_.size() < static_cast<size_t>(num_targets_)) {
        last_altitudes_.resize(num_targets_, 0.0f);
    }

    for (int i = 0; i < num_targets_; i++) {
        const auto& param = target_params_[i];

        // 出現前または消失後はスキップ
        if (time < param.birth_time) continue;
        if (param.death_time > 0.0 && time > param.death_time) continue;

        double dt = time - param.birth_time;
        StateVector state;

        switch (param.motion_model) {
            case MotionModel::CONSTANT_VELOCITY:
                state = propagateConstantVelocity(param.initial_state, dt);
                break;
            case MotionModel::CONSTANT_ACCELERATION:
                state = propagateConstantAcceleration(param.initial_state, dt);
                break;
            case MotionModel::MANEUVERING:
                state = propagateManeuver(param.initial_state, dt,
                                         param.turn_rate, param.accel_magnitude);
                break;
            case MotionModel::BALLISTIC_MISSILE:
                state = propagateBallisticMissile(param, time);
                // 弾道ミサイルの場合、キャッシュから高度を取得
                if (param.trajectory_computed && !param.trajectory_cache.empty()) {
                    double elapsed = time - param.birth_time;
                    // 二分探索で高度を取得
                    int lo = 0, hi = static_cast<int>(param.trajectory_cache.size()) - 1;
                    if (elapsed <= param.trajectory_cache.front().time) {
                        last_altitudes_[i] = param.trajectory_cache.front().z;
                    } else if (elapsed >= param.trajectory_cache.back().time) {
                        last_altitudes_[i] = param.trajectory_cache.back().z;
                    } else {
                        while (lo < hi - 1) {
                            int mid = (lo + hi) / 2;
                            if (param.trajectory_cache[mid].time <= elapsed) lo = mid; else hi = mid;
                        }
                        float frac = static_cast<float>(
                            (elapsed - param.trajectory_cache[lo].time) /
                            (param.trajectory_cache[hi].time - param.trajectory_cache[lo].time));
                        last_altitudes_[i] = param.trajectory_cache[lo].z +
                            frac * (param.trajectory_cache[hi].z - param.trajectory_cache[lo].z);
                    }
                }
                break;
            case MotionModel::HYPERSONIC_GLIDE:
                state = propagateHypersonicGlide(param, time);
                // HGVもキャッシュから高度を取得（弾道ミサイルと同パターン）
                if (param.trajectory_computed && !param.trajectory_cache.empty()) {
                    double elapsed = time - param.birth_time;
                    int lo = 0, hi = static_cast<int>(param.trajectory_cache.size()) - 1;
                    if (elapsed <= param.trajectory_cache.front().time) {
                        last_altitudes_[i] = param.trajectory_cache.front().z;
                    } else if (elapsed >= param.trajectory_cache.back().time) {
                        last_altitudes_[i] = param.trajectory_cache.back().z;
                    } else {
                        while (lo < hi - 1) {
                            int mid = (lo + hi) / 2;
                            if (param.trajectory_cache[mid].time <= elapsed) lo = mid; else hi = mid;
                        }
                        float frac = static_cast<float>(
                            (elapsed - param.trajectory_cache[lo].time) /
                            (param.trajectory_cache[hi].time - param.trajectory_cache[lo].time));
                        last_altitudes_[i] = param.trajectory_cache[lo].z +
                            frac * (param.trajectory_cache[hi].z - param.trajectory_cache[lo].z);
                    }
                }
                break;
        }

        states.push_back(state);
    }

    return states;
}

std::vector<int> TargetGenerator::getActiveTargets(double time) const {
    std::vector<int> active_targets;

    for (int i = 0; i < num_targets_; i++) {
        const auto& param = target_params_[i];
        if (time >= param.birth_time &&
            (param.death_time < 0.0 || time <= param.death_time)) {
            active_targets.push_back(i);
        }
    }

    return active_targets;
}

StateVector TargetGenerator::propagateConstantVelocity(const StateVector& state, double dt) const {
    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    StateVector new_state = StateVector::Zero();
    float t = static_cast<float>(dt);

    new_state(0) = state(0) + state(3) * t;  // x += vx*t
    new_state(1) = state(1) + state(4) * t;  // y += vy*t
    new_state(2) = state(2) + state(5) * t;  // z += vz*t
    new_state(3) = state(3);   // vx
    new_state(4) = state(4);   // vy
    new_state(5) = state(5);   // vz

    return new_state;
}

StateVector TargetGenerator::propagateConstantAcceleration(const StateVector& state, double dt) const {
    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    StateVector new_state = StateVector::Zero();
    float t = static_cast<float>(dt);
    float t2 = 0.5f * t * t;

    new_state(0) = state(0) + state(3) * t + state(6) * t2;  // x += vx*t + 0.5*ax*t²
    new_state(1) = state(1) + state(4) * t + state(7) * t2;  // y += vy*t + 0.5*ay*t²
    new_state(2) = state(2) + state(5) * t + state(8) * t2;  // z += vz*t + 0.5*az*t²
    new_state(3) = state(3) + state(6) * t;   // vx += ax*t
    new_state(4) = state(4) + state(7) * t;   // vy += ay*t
    new_state(5) = state(5) + state(8) * t;   // vz += az*t
    new_state(6) = state(6);   // ax
    new_state(7) = state(7);   // ay
    new_state(8) = state(8);   // az

    return new_state;
}

StateVector TargetGenerator::propagateManeuver(const StateVector& state, double dt,
                                                float turn_rate, float accel_mag) const {
    // 協調旋回モデル（Coordinated Turn Model）
    StateVector new_state = state;
    float t = static_cast<float>(dt);

    // 小さな時間ステップで数値積分
    int steps = static_cast<int>(dt * 10) + 1;  // 0.1秒ステップ
    float dt_step = t / steps;

    for (int i = 0; i < steps; i++) {
        // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
        float vx = new_state(3);
        float vy = new_state(4);

        // 速度の向きに応じた加速度
        float speed = std::sqrt(vx * vx + vy * vy);
        if (speed < 1e-6f) speed = 1e-6f;

        // 接線方向の加速度（前進加速）
        float ax_tangent = accel_mag * (vx / speed);
        float ay_tangent = accel_mag * (vy / speed);

        // 法線方向の加速度（旋回）
        float ax_normal = -turn_rate * vy;
        float ay_normal = turn_rate * vx;

        // 合成加速度
        float ax = ax_tangent + ax_normal;
        float ay = ay_tangent + ay_normal;

        // オイラー法で更新
        new_state(0) += vx * dt_step;
        new_state(1) += vy * dt_step;
        new_state(3) += ax * dt_step;
        new_state(4) += ay * dt_step;
        new_state(6) = ax;
        new_state(7) = ay;
    }

    return new_state;
}

void TargetGenerator::generateClusteredScenario(const Eigen::Vector2f& cluster_center,
                                                 float cluster_radius) {
    std::cout << "Generating clustered scenario at (" << cluster_center.transpose()
              << ") with radius " << cluster_radius << " m" << std::endl;

    for (int i = 0; i < num_targets_; i++) {
        auto& param = target_params_[i];

        // クラスタ内にランダム配置
        float r = cluster_radius * std::sqrt(uniform_dist_(rng_));
        float theta = uniform_dist_(rng_) * 2.0f * M_PI;

        float x = cluster_center(0) + r * std::cos(theta);
        float y = cluster_center(1) + r * std::sin(theta);

        // 速度: クラスタ全体で同じ方向に移動
        float common_heading = uniform_dist_(rng_) * 2.0f * M_PI;
        float speed = 50.0f + uniform_dist_(rng_) * 50.0f;  // 50~100 m/s
        float vx = speed * std::cos(common_heading);
        float vy = speed * std::sin(common_heading);

        param.initial_state << x, y, vx, vy, 0.0f, 0.0f;
        param.motion_model = MotionModel::CONSTANT_VELOCITY;
        param.birth_time = 0.0;
        param.death_time = -1.0;
    }
}

void TargetGenerator::generateHighManeuverScenario() {
    std::cout << "Generating high-maneuver scenario" << std::endl;

    for (int i = 0; i < num_targets_; i++) {
        auto& param = target_params_[i];
        param.initial_state = generateRandomInitialState();
        param.motion_model = MotionModel::MANEUVERING;

        // 高い旋回レートと加速度
        param.turn_rate = (uniform_dist_(rng_) - 0.5f) * 0.2f;  // -0.1~0.1 rad/s
        param.accel_magnitude = 5.0f + uniform_dist_(rng_) * 10.0f;  // 5~15 m/s²
        param.birth_time = 0.0;
        param.death_time = -1.0;
    }

    std::cout << "  All targets with high maneuverability" << std::endl;
}

} // namespace fasttracker
