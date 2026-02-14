// 弾道ミサイル軌道生成（RK4物理モデル）
#include "simulation/target_generator.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace fasttracker {

// ========================================
// 物理モデルヘルパー（ファイルスコープ）
// ========================================

struct PhysicsState {
    float x, y, z;       // 位置 [m]
    float vx, vy, vz;    // 速度 [m/s]
    float mass;           // 質量 [kg]
};

struct PhysicsDerivative {
    float dx, dy, dz;
    float dvx, dvy, dvz;
    float dmass;
};

static float airDensity(float altitude, float rho0, float scale_height) {
    if (altitude < 0.0f) altitude = 0.0f;
    return rho0 * std::exp(-altitude / scale_height);
}

static float gravityAtAltitude(float altitude, float g0, float earth_radius) {
    float r_ratio = earth_radius / (earth_radius + std::max(altitude, 0.0f));
    return g0 * r_ratio * r_ratio;
}

static PhysicsDerivative computeDerivatives(
    const PhysicsState& s,
    double elapsed,
    const MissileParameters& mp,
    float azimuth,
    float g0,
    float earth_radius,
    float rho0,
    float scale_height)
{
    PhysicsDerivative d;
    d.dx = s.vx;
    d.dy = s.vy;
    d.dz = s.vz;
    d.dmass = 0.0f;

    float speed = std::sqrt(s.vx * s.vx + s.vy * s.vy + s.vz * s.vz);

    // 1. 重力（鉛直下向き）
    float g = gravityAtAltitude(s.z, g0, earth_radius);
    float fz_gravity = -s.mass * g;

    // 2. 推力（ブーストフェーズ中）
    float fx_thrust = 0.0f, fy_thrust = 0.0f, fz_thrust = 0.0f;
    float dry_mass = mp.initial_mass * (1.0f - mp.fuel_fraction);

    if (elapsed < mp.boost_duration && s.mass > dry_mass) {
        float mdot = mp.initial_mass * mp.fuel_fraction / mp.boost_duration;
        float thrust = mp.specific_impulse * g0 * mdot;
        d.dmass = -mdot;

        // ピッチプログラム: ブースト中は固定方向で推力を発生
        // 重力ターンはブースト後半で徐々に導入
        float pitch_program_end = mp.boost_duration * 0.6f;  // ブーストの60%まで固定方向

        // 固定方向成分
        float fx_fixed = thrust * std::cos(mp.launch_angle) * std::cos(azimuth);
        float fy_fixed = thrust * std::cos(mp.launch_angle) * std::sin(azimuth);
        float fz_fixed = thrust * std::sin(mp.launch_angle);

        if (elapsed < pitch_program_end || speed < 1.0f) {
            // ピッチプログラム中: 固定方向
            fx_thrust = fx_fixed;
            fy_thrust = fy_fixed;
            fz_thrust = fz_fixed;
        } else {
            // ブースト後半: 固定方向から重力ターンへ徐々に遷移
            float blend = (elapsed - pitch_program_end) / (mp.boost_duration - pitch_program_end);
            blend = std::min(blend, 1.0f);
            blend = blend * blend;  // 二乗関数で滑らかに遷移

            float fx_gt = thrust * (s.vx / speed);
            float fy_gt = thrust * (s.vy / speed);
            float fz_gt = thrust * (s.vz / speed);

            fx_thrust = (1.0f - blend) * fx_fixed + blend * fx_gt;
            fy_thrust = (1.0f - blend) * fy_fixed + blend * fy_gt;
            fz_thrust = (1.0f - blend) * fz_fixed + blend * fz_gt;
        }
    }

    // 3. 大気抗力
    float fx_drag = 0.0f, fy_drag = 0.0f, fz_drag = 0.0f;
    if (s.z >= 0.0f && speed > 0.01f) {
        float rho = airDensity(s.z, rho0, scale_height);
        float drag = 0.5f * rho * speed * speed * mp.drag_coefficient * mp.cross_section_area;
        fx_drag = -drag * (s.vx / speed);
        fy_drag = -drag * (s.vy / speed);
        fz_drag = -drag * (s.vz / speed);
    }

    // 合計加速度 = F/m
    float inv_mass = 1.0f / std::max(s.mass, 1.0f);
    d.dvx = (fx_thrust + fx_drag) * inv_mass;
    d.dvy = (fy_thrust + fy_drag) * inv_mass;
    d.dvz = (fz_thrust + fz_gravity + fz_drag) * inv_mass;

    return d;
}

static PhysicsState addScaled(const PhysicsState& s, const PhysicsDerivative& d, float h) {
    PhysicsState r;
    r.x = s.x + d.dx * h;
    r.y = s.y + d.dy * h;
    r.z = s.z + d.dz * h;
    r.vx = s.vx + d.dvx * h;
    r.vy = s.vy + d.dvy * h;
    r.vz = s.vz + d.dvz * h;
    r.mass = s.mass + d.dmass * h;
    return r;
}

static PhysicsState rk4Step(
    const PhysicsState& state,
    double elapsed,
    float dt,
    const MissileParameters& mp,
    float azimuth,
    float g0, float earth_radius, float rho0, float scale_height)
{
    auto k1 = computeDerivatives(state, elapsed, mp, azimuth, g0, earth_radius, rho0, scale_height);
    auto s2 = addScaled(state, k1, 0.5f * dt);
    auto k2 = computeDerivatives(s2, elapsed + 0.5 * dt, mp, azimuth, g0, earth_radius, rho0, scale_height);
    auto s3 = addScaled(state, k2, 0.5f * dt);
    auto k3 = computeDerivatives(s3, elapsed + 0.5 * dt, mp, azimuth, g0, earth_radius, rho0, scale_height);
    auto s4 = addScaled(state, k3, dt);
    auto k4 = computeDerivatives(s4, elapsed + dt, mp, azimuth, g0, earth_radius, rho0, scale_height);

    PhysicsState result;
    float h6 = dt / 6.0f;
    result.x    = state.x    + h6 * (k1.dx  + 2*k2.dx  + 2*k3.dx  + k4.dx);
    result.y    = state.y    + h6 * (k1.dy  + 2*k2.dy  + 2*k3.dy  + k4.dy);
    result.z    = state.z    + h6 * (k1.dz  + 2*k2.dz  + 2*k3.dz  + k4.dz);
    result.vx   = state.vx   + h6 * (k1.dvx + 2*k2.dvx + 2*k3.dvx + k4.dvx);
    result.vy   = state.vy   + h6 * (k1.dvy + 2*k2.dvy + 2*k3.dvy + k4.dvy);
    result.vz   = state.vz   + h6 * (k1.dvz + 2*k2.dvz + 2*k3.dvz + k4.dvz);
    result.mass = state.mass + h6 * (k1.dmass + 2*k2.dmass + 2*k3.dmass + k4.dmass);

    // 高度クランプ
    if (result.z < 0.0f) result.z = 0.0f;

    // 質量クランプ
    float dry_mass = mp.initial_mass * (1.0f - mp.fuel_fraction);
    if (result.mass < dry_mass) result.mass = dry_mass;

    return result;
}

// ========================================
// 軌道キャッシュからの補間
// ========================================

static TrajectoryPoint interpolateCache(
    const std::vector<TrajectoryPoint>& cache, double elapsed)
{
    if (cache.empty()) {
        return TrajectoryPoint{0, 0, 0, 0, 0, 0, 0, 0, BallisticPhase::BOOST};
    }
    if (elapsed <= cache.front().time) return cache.front();
    if (elapsed >= cache.back().time) return cache.back();

    // 二分探索
    int lo = 0, hi = static_cast<int>(cache.size()) - 1;
    while (lo < hi - 1) {
        int mid = (lo + hi) / 2;
        if (cache[mid].time <= elapsed) lo = mid; else hi = mid;
    }

    const auto& a = cache[lo];
    const auto& b = cache[hi];
    float frac = (b.time > a.time) ?
        static_cast<float>((elapsed - a.time) / (b.time - a.time)) : 0.0f;

    TrajectoryPoint p;
    p.time = elapsed;
    p.x  = a.x  + frac * (b.x  - a.x);
    p.y  = a.y  + frac * (b.y  - a.y);
    p.z  = a.z  + frac * (b.z  - a.z);
    p.vx = a.vx + frac * (b.vx - a.vx);
    p.vy = a.vy + frac * (b.vy - a.vy);
    p.vz = a.vz + frac * (b.vz - a.vz);
    p.mass = a.mass + frac * (b.mass - a.mass);
    p.phase = (frac < 0.5f) ? a.phase : b.phase;

    return p;
}

// ========================================
// 弾道軌道の事前計算（RK4物理積分）
// ========================================

void TargetGenerator::precomputeBallisticTrajectory(TargetParameters& params, bool verbose) const {
    const auto& mp = params.missile_params;
    params.trajectory_cache.clear();

    float x0 = params.initial_state(0);
    float y0 = params.initial_state(1);
    float xt = mp.target_position(0);
    float yt = mp.target_position(1);
    float azimuth = std::atan2(yt - y0, xt - x0);

    // 初期3D状態
    PhysicsState state;
    state.x = x0;
    state.y = y0;
    state.z = 0.0f;
    state.vx = 0.0f;
    state.vy = 0.0f;
    state.vz = 0.0f;
    state.mass = mp.initial_mass;

    float dt_physics = 0.05f;  // 50ms精度
    double max_time = 3600.0;  // 最大1時間（安全上限）
    bool past_apogee = false;

    for (double t = 0.0; t <= max_time; t += dt_physics) {
        // フェーズ判定
        BallisticPhase phase;
        if (t < mp.boost_duration) {
            phase = BallisticPhase::BOOST;
        } else if (state.vz > 0.0f || state.z > 50000.0f) {
            phase = BallisticPhase::MIDCOURSE;
        } else {
            phase = BallisticPhase::TERMINAL;
            past_apogee = true;
        }

        // キャッシュに保存
        TrajectoryPoint tp;
        tp.time = t;
        tp.x = state.x;
        tp.y = state.y;
        tp.z = state.z;
        tp.vx = state.vx;
        tp.vy = state.vy;
        tp.vz = state.vz;
        tp.mass = state.mass;
        tp.phase = phase;
        params.trajectory_cache.push_back(tp);

        // RK4ステップ
        state = rk4Step(state, t, dt_physics, mp, azimuth,
                        GRAVITY, EARTH_RADIUS, RHO0, SCALE_HEIGHT);

        // 着弾判定: ブースト後に高度≤0
        if (t > mp.boost_duration && state.z <= 0.0f) {
            // 最終点（着弾）
            state.z = 0.0f;
            TrajectoryPoint impact;
            impact.time = t + dt_physics;
            impact.x = state.x;
            impact.y = state.y;
            impact.z = 0.0f;
            impact.vx = state.vx;
            impact.vy = state.vy;
            impact.vz = state.vz;
            impact.mass = state.mass;
            impact.phase = BallisticPhase::TERMINAL;
            params.trajectory_cache.push_back(impact);

            // death_timeを実際の着弾時刻に更新
            params.death_time = params.birth_time + t + dt_physics;
            break;
        }
    }

    params.trajectory_computed = true;

    // デバッグ出力
    if (verbose && !params.trajectory_cache.empty()) {
        float max_alt = 0.0f;
        float max_speed = 0.0f;
        for (const auto& tp : params.trajectory_cache) {
            max_alt = std::max(max_alt, tp.z);
            float spd = std::sqrt(tp.vx*tp.vx + tp.vy*tp.vy + tp.vz*tp.vz);
            max_speed = std::max(max_speed, spd);
        }
        std::cerr << "  Trajectory computed: "
                  << params.trajectory_cache.size() << " points, "
                  << "flight_time=" << (params.trajectory_cache.back().time) << "s, "
                  << "max_alt=" << (max_alt / 1000.0f) << "km, "
                  << "max_speed=" << max_speed << "m/s ("
                  << (max_speed / 340.0f) << " Mach)" << std::endl;
    }
}

// ========================================
// 高度アクセサ
// ========================================

float TargetGenerator::getLastAltitude(int target_idx) const {
    if (target_idx >= 0 && target_idx < static_cast<int>(last_altitudes_.size())) {
        return last_altitudes_[target_idx];
    }
    return 0.0f;
}

// ========================================
// フェーズ判定（キャッシュベース）
// ========================================

BallisticPhase TargetGenerator::getBallisticPhase(const TargetParameters& params, double time) const {
    double elapsed = time - params.birth_time;

    if (params.trajectory_computed && !params.trajectory_cache.empty()) {
        auto tp = interpolateCache(params.trajectory_cache, elapsed);
        return tp.phase;
    }

    // フォールバック: 時間割合ベース
    if (elapsed < params.missile_params.boost_duration) {
        return BallisticPhase::BOOST;
    }
    double total_flight = params.death_time - params.birth_time;
    double midcourse_end = params.birth_time + total_flight * 0.8;
    if (time < midcourse_end) {
        return BallisticPhase::MIDCOURSE;
    }
    return BallisticPhase::TERMINAL;
}

// ========================================
// 弾道ミサイル軌道（キャッシュ補間）
// ========================================

StateVector TargetGenerator::propagateBallisticMissile(const TargetParameters& params, double time) const {
    StateVector state;
    state.setZero();
    double elapsed = time - params.birth_time;

    if (elapsed < 0) return state;

    // 初回呼び出し時にRK4で事前計算
    if (!params.trajectory_computed) {
        const_cast<TargetGenerator*>(this)->precomputeBallisticTrajectory(
            const_cast<TargetParameters&>(params));
    }

    if (params.trajectory_cache.empty()) return state;

    // キャッシュから補間
    auto tp = interpolateCache(params.trajectory_cache, elapsed);

    state(0) = tp.x;
    state(1) = tp.y;
    state(2) = tp.vx;
    state(3) = tp.vy;

    // 加速度は有限差分で計算
    double dt_fd = 0.1;
    auto tp_prev = interpolateCache(params.trajectory_cache, std::max(0.0, elapsed - dt_fd));
    state(4) = (tp.vx - tp_prev.vx) / static_cast<float>(dt_fd);
    state(5) = (tp.vy - tp_prev.vy) / static_cast<float>(dt_fd);

    // 高度を保存（外部から参照用）
    // target_idxを直接知らないので、last_altitudes_をresizeして格納
    // generateStates()側で target_idx を使って保存する

    return state;
}

// ========================================
// シナリオ生成関数（既存維持）
// ========================================

void TargetGenerator::generateBallisticMissileScenario(int num_missiles) {
    target_params_.clear();
    num_targets_ = num_missiles;

    for (int i = 0; i < num_missiles; i++) {
        TargetParameters params;
        params.motion_model = MotionModel::BALLISTIC_MISSILE;
        params.birth_time = i * 5.0;

        float launch_range = 50000.0f + uniform_dist_(rng_) * 50000.0f;
        float launch_azimuth = (uniform_dist_(rng_) - 0.5f) * M_PI;

        params.initial_state(0) = launch_range * std::cos(launch_azimuth);
        params.initial_state(1) = launch_range * std::sin(launch_azimuth);
        params.initial_state(2) = 0.0f;
        params.initial_state(3) = 0.0f;

        params.missile_params.launch_angle = 0.6f + uniform_dist_(rng_) * 0.4f;
        params.missile_params.max_altitude = 30000.0f + uniform_dist_(rng_) * 40000.0f;
        params.missile_params.boost_duration = 40.0f + uniform_dist_(rng_) * 40.0f;
        params.missile_params.boost_acceleration = 15.0f + uniform_dist_(rng_) * 10.0f;

        float impact_range = 5000.0f + uniform_dist_(rng_) * 10000.0f;
        float impact_azimuth = (uniform_dist_(rng_) - 0.5f) * 0.5f;
        params.missile_params.target_position(0) = impact_range * std::cos(impact_azimuth);
        params.missile_params.target_position(1) = impact_range * std::sin(impact_azimuth);

        // death_timeは暫定（precomputeで更新される）
        params.death_time = params.birth_time + 600.0;

        target_params_.push_back(params);
    }
}

void TargetGenerator::generateHypersonicGlideScenario(int num_hgvs) {
    target_params_.clear();
    num_targets_ = num_hgvs;

    for (int i = 0; i < num_hgvs; i++) {
        TargetParameters params;
        params.motion_model = MotionModel::HYPERSONIC_GLIDE;
        params.birth_time = i * 10.0;

        float initial_range = 100000.0f + uniform_dist_(rng_) * 50000.0f;
        float initial_azimuth = (uniform_dist_(rng_) - 0.5f) * M_PI;

        params.initial_state(0) = initial_range * std::cos(initial_azimuth);
        params.initial_state(1) = initial_range * std::sin(initial_azimuth);

        float initial_speed = 1500.0f + uniform_dist_(rng_) * 2000.0f;
        float velocity_azimuth = initial_azimuth + M_PI;

        params.initial_state(2) = initial_speed * std::cos(velocity_azimuth);
        params.initial_state(3) = initial_speed * std::sin(velocity_azimuth);

        params.missile_params.cruise_altitude = 20000.0f + uniform_dist_(rng_) * 20000.0f;
        params.missile_params.glide_ratio = 3.0f + uniform_dist_(rng_) * 3.0f;
        params.missile_params.skip_amplitude = 3000.0f + uniform_dist_(rng_) * 4000.0f;
        params.missile_params.skip_frequency = 0.005f + uniform_dist_(rng_) * 0.015f;
        params.missile_params.terminal_dive_range = 10000.0f + uniform_dist_(rng_) * 15000.0f;

        float target_range = 5000.0f + uniform_dist_(rng_) * 10000.0f;
        float target_azimuth = (uniform_dist_(rng_) - 0.5f) * 0.3f;
        params.missile_params.target_position(0) = target_range * std::cos(target_azimuth);
        params.missile_params.target_position(1) = target_range * std::sin(target_azimuth);

        params.death_time = params.birth_time + 60.0 + uniform_dist_(rng_) * 60.0;

        target_params_.push_back(params);
    }
}

void TargetGenerator::generateMixedThreatScenario() {
    target_params_.clear();
    int num_aircraft = num_targets_ / 2;
    int num_ballistic = num_targets_ / 4;
    int num_hgv = num_targets_ - num_aircraft - num_ballistic;

    initializeDefaultScenario();
    auto aircraft_params = target_params_;

    generateBallisticMissileScenario(num_ballistic);
    auto ballistic_params = target_params_;

    generateHypersonicGlideScenario(num_hgv);
    auto hgv_params = target_params_;

    target_params_.clear();
    target_params_.insert(target_params_.end(), aircraft_params.begin(), aircraft_params.end());
    target_params_.insert(target_params_.end(), ballistic_params.begin(), ballistic_params.end());
    target_params_.insert(target_params_.end(), hgv_params.begin(), hgv_params.end());

    num_targets_ = static_cast<int>(target_params_.size());
}

StateVector TargetGenerator::propagateHypersonicGlide(const TargetParameters& params, double time) const {
    StateVector state = params.initial_state;
    double elapsed = time - params.birth_time;

    if (elapsed < 0) {
        state.setZero();
        return state;
    }

    float x0 = params.initial_state(0);
    float y0 = params.initial_state(1);
    float vx0 = params.initial_state(2);
    float vy0 = params.initial_state(3);

    float xt = params.missile_params.target_position(0);
    float yt = params.missile_params.target_position(1);

    float dx = xt - x0;
    float dy = yt - y0;
    float range_to_target = std::sqrt(dx * dx + dy * dy);
    float azimuth_to_target = std::atan2(dy, dx);

    float cruise_speed = std::sqrt(vx0 * vx0 + vy0 * vy0);
    float current_range = range_to_target * (1.0f - elapsed / (params.death_time - params.birth_time));

    if (current_range > params.missile_params.terminal_dive_range) {
        float t = elapsed;
        float progress = elapsed / (params.death_time - params.birth_time);
        float base_x = x0 + (xt - x0) * progress;
        float base_y = y0 + (yt - y0) * progress;

        float skip_phase = 2.0f * M_PI * params.missile_params.skip_frequency * t;
        float skip_offset_x = params.missile_params.skip_amplitude * std::sin(skip_phase) *
                             std::cos(azimuth_to_target + M_PI / 2.0f);
        float skip_offset_y = params.missile_params.skip_amplitude * std::sin(skip_phase) *
                             std::sin(azimuth_to_target + M_PI / 2.0f);

        state(0) = base_x + skip_offset_x;
        state(1) = base_y + skip_offset_y;

        float vx_cruise = cruise_speed * std::cos(azimuth_to_target);
        float vy_cruise = cruise_speed * std::sin(azimuth_to_target);

        float skip_vel_scale = std::cos(skip_phase);
        state(2) = vx_cruise * (1.0f + 0.2f * skip_vel_scale);
        state(3) = vy_cruise * (1.0f + 0.2f * skip_vel_scale);

        state(4) = -50.0f * std::sin(skip_phase) * std::cos(azimuth_to_target + M_PI / 2.0f);
        state(5) = -50.0f * std::sin(skip_phase) * std::sin(azimuth_to_target + M_PI / 2.0f);
    } else {
        float terminal_progress = (params.missile_params.terminal_dive_range - current_range) /
                                 params.missile_params.terminal_dive_range;

        float dive_speed = cruise_speed * (1.0f + terminal_progress * 0.5f);

        state(0) = xt - (xt - x0) * (1.0f - elapsed / (params.death_time - params.birth_time));
        state(1) = yt - (yt - y0) * (1.0f - elapsed / (params.death_time - params.birth_time));
        state(2) = dive_speed * std::cos(azimuth_to_target);
        state(3) = dive_speed * std::sin(azimuth_to_target);

        float maneuver_freq = 3.0f;
        state(4) = 80.0f * std::sin(maneuver_freq * elapsed);
        state(5) = 80.0f * std::cos(maneuver_freq * elapsed * 1.3f);
    }

    return state;
}

} // namespace fasttracker
