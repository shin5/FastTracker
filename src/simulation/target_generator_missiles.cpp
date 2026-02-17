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
    params.booster_trajectory_cache.clear();

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

    // 分離イベント記録用
    PhysicsState separation_state;
    double separation_time = -1.0;
    bool separated = false;

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
        tp.object_id = 0;
        params.trajectory_cache.push_back(tp);

        // 分離判定: ブースト終了の瞬間に分離イベント発生
        if (mp.enable_separation && !separated && t >= mp.boost_duration) {
            separated = true;
            separation_state = state;
            separation_time = t;

            // 弾頭の質量を変更: dry_mass * warhead_mass_fraction
            float dry_mass = mp.initial_mass * (1.0f - mp.fuel_fraction);
            state.mass = dry_mass * mp.warhead_mass_fraction;

            if (verbose) {
                std::cerr << "  Separation at t=" << t << "s: "
                          << "warhead_mass=" << state.mass << "kg, "
                          << "alt=" << (state.z / 1000.0f) << "km" << std::endl;
            }
        }

        // RK4ステップ（分離後の弾頭は warhead_cd を使用）
        MissileParameters mp_step = mp;
        if (separated) {
            mp_step.drag_coefficient = mp.warhead_cd;
        }
        state = rk4Step(state, t, dt_physics, mp_step, azimuth,
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
            impact.object_id = 0;
            params.trajectory_cache.push_back(impact);

            // death_timeを実際の着弾時刻に更新
            params.death_time = params.birth_time + t + dt_physics;
            break;
        }
    }

    // ブースター軌道の生成（分離が有効で分離イベントが発生した場合）
    if (mp.enable_separation && separated) {
        float dry_mass = mp.initial_mass * (1.0f - mp.fuel_fraction);
        float booster_mass = dry_mass * (1.0f - mp.warhead_mass_fraction);

        PhysicsState bs = separation_state;
        bs.mass = booster_mass;

        // ブースター用のMissileParameters（推力なし、高Cd）
        MissileParameters mp_booster = mp;
        mp_booster.boost_duration = 0.0f;  // 推力なし（既にブースト終了）
        mp_booster.drag_coefficient = mp.booster_cd;

        for (double t = separation_time; t <= max_time; t += dt_physics) {
            TrajectoryPoint btp;
            btp.time = t;
            btp.x = bs.x;
            btp.y = bs.y;
            btp.z = bs.z;
            btp.vx = bs.vx;
            btp.vy = bs.vy;
            btp.vz = bs.vz;
            btp.mass = bs.mass;
            btp.phase = (bs.vz > 0.0f) ? BallisticPhase::MIDCOURSE : BallisticPhase::TERMINAL;
            btp.object_id = 1;
            params.booster_trajectory_cache.push_back(btp);

            bs = rk4Step(bs, t, dt_physics, mp_booster, azimuth,
                         GRAVITY, EARTH_RADIUS, RHO0, SCALE_HEIGHT);

            // ブースター着弾判定
            if (bs.z <= 0.0f) {
                bs.z = 0.0f;
                TrajectoryPoint bimpact;
                bimpact.time = t + dt_physics;
                bimpact.x = bs.x;
                bimpact.y = bs.y;
                bimpact.z = 0.0f;
                bimpact.vx = bs.vx;
                bimpact.vy = bs.vy;
                bimpact.vz = bs.vz;
                bimpact.mass = bs.mass;
                bimpact.phase = BallisticPhase::TERMINAL;
                bimpact.object_id = 1;
                params.booster_trajectory_cache.push_back(bimpact);
                break;
            }
        }

        if (verbose) {
            std::cerr << "  Booster trajectory: "
                      << params.booster_trajectory_cache.size() << " points, "
                      << "mass=" << booster_mass << "kg, Cd=" << mp.booster_cd << std::endl;
            if (!params.booster_trajectory_cache.empty()) {
                const auto& blast = params.booster_trajectory_cache.back();
                std::cerr << "  Booster impact at t=" << blast.time << "s" << std::endl;
            }
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
// HGV物理モデル（揚力・バンク角誘導）
// ========================================

static float computeBankAngle(
    const PhysicsState& s,
    const MissileParameters& mp,
    float azimuth_to_target,
    float distance_to_target,
    double elapsed,
    BallisticPhase phase,
    float rho0 = 1.225f, float scale_height = 7400.0f,
    float g0 = 9.81f, float earth_radius = 6371000.0f)
{
    if (phase == BallisticPhase::BOOST) {
        return 0.0f;
    }

    if (phase == BallisticPhase::PULLUP) {
        // PULLUP: 物理ベースのバンク角制御
        // 目標: gammaを0に減衰させる（飛翔経路角を水平に遷移）
        // 低高度ではL >> mg なので、σ=0だとgammaが増加してしまう
        // → 鉛直揚力成分 = L·cosσ が目標荷重倍率×mg·cosγ になるようσを計算
        float speed = std::sqrt(s.vx*s.vx + s.vy*s.vy + s.vz*s.vz);
        float speed_h = std::sqrt(s.vx*s.vx + s.vy*s.vy);
        float gamma = std::atan2(s.vz, std::max(speed_h, 0.1f));

        float rho = rho0 * std::exp(-std::max(s.z, 0.0f) / scale_height);
        float r_ratio = earth_radius / (earth_radius + std::max(s.z, 0.0f));
        float g = g0 * r_ratio * r_ratio;

        float Cl = mp.drag_coefficient * mp.glide_ratio;
        float L = 0.5f * rho * speed * speed * Cl * mp.cross_section_area;
        float W_perp = s.mass * g * std::cos(gamma);  // 重力の速度法線成分

        if (L > 1.0f) {
            // cosσ_level: 等γ飛行に必要なバンク角（L·cosσ = mg·cosγ）
            float cos_sigma_level = std::min(1.0f, W_perp / L);
            // gamma補正: gamma > 0 → バンクをさらに増やして鉛直揚力を減らす
            // gamma ≈ 0 → 補正なし（水平維持）
            // gamma < 0 → バンクを減らして引き起こし
            float gamma_correction = -1.5f * gamma;
            float cos_sigma = std::max(-1.0f, std::min(1.0f, cos_sigma_level + gamma_correction));
            float sigma_mag = std::acos(cos_sigma);

            // 方位維持: heading errorに基づいてバンク角に符号を付与
            // 符号なしだと常に同方向に横力が作用し、系統的なドリフトが発生する
            float current_heading = std::atan2(s.vy, s.vx);
            float heading_error = azimuth_to_target - current_heading;
            while (heading_error > static_cast<float>(M_PI)) heading_error -= 2.0f * static_cast<float>(M_PI);
            while (heading_error < -static_cast<float>(M_PI)) heading_error += 2.0f * static_cast<float>(M_PI);
            // heading_error > 0 → 目標は左 → 左バンク(σ < 0)
            return (heading_error >= 0.0f) ? -sigma_mag : sigma_mag;
        }
        return 0.0f;
    }

    float speed_h = std::sqrt(s.vx * s.vx + s.vy * s.vy);
    float current_heading = std::atan2(s.vy, s.vx);
    float heading_error = azimuth_to_target - current_heading;
    // 角度正規化
    while (heading_error > M_PI) heading_error -= 2.0f * M_PI;
    while (heading_error < -M_PI) heading_error += 2.0f * M_PI;

    if (phase == BallisticPhase::GLIDE) {
        // 方位制御のみ: バンク角で目標方向へ誘導
        // 高度制御は不要 — skip-glide振動は揚力/重力/大気密度のバランスから自然に発生
        // 符号: heading_error > 0 → 目標は反時計回り → 左バンク(σ < 0) → 反時計回り旋回
        float K_hdg = 2.0f;
        float sigma = std::max(-mp.bank_angle_max,
                      std::min(mp.bank_angle_max, -K_hdg * heading_error));
        return sigma;
    }

    if (phase == BallisticPhase::TERMINAL) {
        // 終末ダイブ: σ=90°をベースに（cos90°=0 → 鉛直揚力ゼロ → 重力でダイブ）
        float sigma_dive = static_cast<float>(M_PI) / 2.0f;
        // 方位補正（符号反転: heading_error > 0 → 左バンク → σ_heading < 0）
        float sigma_heading = std::max(-0.4f, std::min(0.4f, -2.0f * heading_error));
        // 回避機動: 振幅を半分に（ダイブ中は過大な機動を避ける）
        float t = static_cast<float>(elapsed);
        float sigma_evasive = mp.terminal_maneuver_amp * 0.3f *
            std::sin(2.0f * M_PI * mp.terminal_maneuver_freq * t +
                     std::cos(3.7f * t) * 0.5f);
        float sigma = sigma_dive + sigma_heading + sigma_evasive;
        return std::max(0.0f, std::min(static_cast<float>(M_PI), sigma));
    }

    return 0.0f;
}

static PhysicsDerivative computeHgvDerivatives(
    const PhysicsState& s,
    double elapsed,
    const MissileParameters& mp,
    float launch_azimuth,
    float azimuth_to_target,
    float distance_to_target,
    float g0, float earth_radius, float rho0, float scale_height,
    BallisticPhase phase)
{
    PhysicsDerivative d;
    d.dx = s.vx;
    d.dy = s.vy;
    d.dz = s.vz;
    d.dmass = 0.0f;

    float speed = std::sqrt(s.vx * s.vx + s.vy * s.vy + s.vz * s.vz);
    float g = gravityAtAltitude(s.z, g0, earth_radius);

    // 1. 重力
    float fz_gravity = -s.mass * g;

    // 2. 推力（ブーストフェーズのみ — 弾道ミサイルと共通ロジック）
    float fx_thrust = 0.0f, fy_thrust = 0.0f, fz_thrust = 0.0f;
    float dry_mass = mp.initial_mass * (1.0f - mp.fuel_fraction);

    if (phase == BallisticPhase::BOOST && elapsed < mp.boost_duration && s.mass > dry_mass) {
        float mdot = mp.initial_mass * mp.fuel_fraction / mp.boost_duration;
        float thrust = mp.specific_impulse * g0 * mdot;
        d.dmass = -mdot;

        float pitch_program_end = mp.boost_duration * 0.6f;
        float fx_fixed = thrust * std::cos(mp.launch_angle) * std::cos(launch_azimuth);
        float fy_fixed = thrust * std::cos(mp.launch_angle) * std::sin(launch_azimuth);
        float fz_fixed = thrust * std::sin(mp.launch_angle);

        if (elapsed < pitch_program_end || speed < 1.0f) {
            fx_thrust = fx_fixed;
            fy_thrust = fy_fixed;
            fz_thrust = fz_fixed;
        } else {
            float blend = static_cast<float>(elapsed - pitch_program_end) /
                          (mp.boost_duration - pitch_program_end);
            blend = std::min(blend, 1.0f);
            blend = blend * blend;
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
    float rho = 0.0f;
    if (s.z >= 0.0f && speed > 0.01f) {
        rho = airDensity(s.z, rho0, scale_height);
        float drag = 0.5f * rho * speed * speed * mp.drag_coefficient * mp.cross_section_area;
        fx_drag = -drag * (s.vx / speed);
        fy_drag = -drag * (s.vy / speed);
        fz_drag = -drag * (s.vz / speed);
    }

    // 4. 揚力（ブースト後のみ）
    float fx_lift = 0.0f, fy_lift = 0.0f, fz_lift = 0.0f;
    if (phase != BallisticPhase::BOOST && speed > 1.0f && rho > 1e-10f) {
        float Cl = mp.drag_coefficient * mp.glide_ratio;
        float lift_mag = 0.5f * rho * speed * speed * Cl * mp.cross_section_area;

        // 揚力方向の計算
        // v̂ = velocity unit vector
        float vx_hat = s.vx / speed;
        float vy_hat = s.vy / speed;
        float vz_hat = s.vz / speed;

        // n̂_up: (0,0,1) の速度方向に垂直な成分を正規化
        // v_perp = up - dot(up, v̂) * v̂
        float dot_up_v = vz_hat;  // dot((0,0,1), v̂)
        float perp_x = -dot_up_v * vx_hat;
        float perp_y = -dot_up_v * vy_hat;
        float perp_z = 1.0f - dot_up_v * vz_hat;
        float perp_mag = std::sqrt(perp_x*perp_x + perp_y*perp_y + perp_z*perp_z);

        float n_up_x, n_up_y, n_up_z;
        if (perp_mag > 1e-6f) {
            n_up_x = perp_x / perp_mag;
            n_up_y = perp_y / perp_mag;
            n_up_z = perp_z / perp_mag;
        } else {
            // 速度がほぼ鉛直: 水平方向をフォールバック
            float speed_h = std::sqrt(s.vx*s.vx + s.vy*s.vy);
            if (speed_h > 1e-4f) {
                n_up_x = -s.vx / speed_h;
                n_up_y = -s.vy / speed_h;
                n_up_z = 0.0f;
            } else {
                n_up_x = 1.0f; n_up_y = 0.0f; n_up_z = 0.0f;
            }
        }

        // n̂_lateral = v̂ × n̂_up
        float n_lat_x = vy_hat * n_up_z - vz_hat * n_up_y;
        float n_lat_y = vz_hat * n_up_x - vx_hat * n_up_z;
        float n_lat_z = vx_hat * n_up_y - vy_hat * n_up_x;

        // バンク角
        float sigma = computeBankAngle(s, mp, azimuth_to_target,
                                        distance_to_target, elapsed, phase,
                                        rho0, scale_height, g0, earth_radius);
        float cos_s = std::cos(sigma);
        float sin_s = std::sin(sigma);

        // lift_dir = cos(σ)·n̂_up + sin(σ)·n̂_lateral
        float ld_x = cos_s * n_up_x + sin_s * n_lat_x;
        float ld_y = cos_s * n_up_y + sin_s * n_lat_y;
        float ld_z = cos_s * n_up_z + sin_s * n_lat_z;

        fx_lift = lift_mag * ld_x;
        fy_lift = lift_mag * ld_y;
        fz_lift = lift_mag * ld_z;
    }

    // 合計加速度
    float inv_mass = 1.0f / std::max(s.mass, 1.0f);
    d.dvx = (fx_thrust + fx_drag + fx_lift) * inv_mass;
    d.dvy = (fy_thrust + fy_drag + fy_lift) * inv_mass;
    d.dvz = (fz_thrust + fz_gravity + fz_drag + fz_lift) * inv_mass;

    return d;
}

static PhysicsState rk4StepHgv(
    const PhysicsState& state, double elapsed, float dt,
    const MissileParameters& mp,
    float launch_azimuth, float azimuth_to_target, float distance_to_target,
    float g0, float earth_radius, float rho0, float scale_height,
    BallisticPhase phase)
{
    auto k1 = computeHgvDerivatives(state, elapsed, mp, launch_azimuth,
        azimuth_to_target, distance_to_target, g0, earth_radius, rho0, scale_height, phase);
    auto s2 = addScaled(state, k1, 0.5f * dt);
    auto k2 = computeHgvDerivatives(s2, elapsed + 0.5*dt, mp, launch_azimuth,
        azimuth_to_target, distance_to_target, g0, earth_radius, rho0, scale_height, phase);
    auto s3 = addScaled(state, k2, 0.5f * dt);
    auto k3 = computeHgvDerivatives(s3, elapsed + 0.5*dt, mp, launch_azimuth,
        azimuth_to_target, distance_to_target, g0, earth_radius, rho0, scale_height, phase);
    auto s4 = addScaled(state, k3, dt);
    auto k4 = computeHgvDerivatives(s4, elapsed + dt, mp, launch_azimuth,
        azimuth_to_target, distance_to_target, g0, earth_radius, rho0, scale_height, phase);

    PhysicsState result;
    float h6 = dt / 6.0f;
    result.x    = state.x    + h6 * (k1.dx  + 2*k2.dx  + 2*k3.dx  + k4.dx);
    result.y    = state.y    + h6 * (k1.dy  + 2*k2.dy  + 2*k3.dy  + k4.dy);
    result.z    = state.z    + h6 * (k1.dz  + 2*k2.dz  + 2*k3.dz  + k4.dz);
    result.vx   = state.vx   + h6 * (k1.dvx + 2*k2.dvx + 2*k3.dvx + k4.dvx);
    result.vy   = state.vy   + h6 * (k1.dvy + 2*k2.dvy + 2*k3.dvy + k4.dvy);
    result.vz   = state.vz   + h6 * (k1.dvz + 2*k2.dvz + 2*k3.dvz + k4.dvz);
    result.mass = state.mass + h6 * (k1.dmass + 2*k2.dmass + 2*k3.dmass + k4.dmass);

    if (result.z < 0.0f) result.z = 0.0f;
    float dry_mass = mp.initial_mass * (1.0f - mp.fuel_fraction);
    if (result.mass < dry_mass) result.mass = dry_mass;

    return result;
}

// ========================================
// HGV軌道の事前計算（RK4物理積分）
// ========================================

void TargetGenerator::precomputeHgvTrajectory(TargetParameters& params, bool verbose) const {
    const auto& mp = params.missile_params;
    params.trajectory_cache.clear();
    params.booster_trajectory_cache.clear();

    float x0 = params.initial_state(0);
    float y0 = params.initial_state(1);
    float xt = mp.target_position(0);
    float yt = mp.target_position(1);
    float launch_azimuth = std::atan2(yt - y0, xt - x0);

    PhysicsState state;
    state.x = x0; state.y = y0; state.z = 0.0f;
    state.vx = 0.0f; state.vy = 0.0f; state.vz = 0.0f;
    state.mass = mp.initial_mass;

    float dt_physics = 0.05f;
    double max_time = 7200.0;

    BallisticPhase phase = BallisticPhase::BOOST;
    bool boost_ended = false;
    double pullup_start_time = 0.0;
    int skip_count = 0;           // 現在のスキップ回数
    bool prev_vz_positive = false; // 前ステップでvz > 0だったか（スキップ検出用）

    for (double t = 0.0; t <= max_time; t += dt_physics) {
        float speed_h = std::sqrt(state.vx*state.vx + state.vy*state.vy);
        float gamma = std::atan2(state.vz, std::max(speed_h, 0.1f));

        // フェーズ遷移
        if (phase == BallisticPhase::BOOST) {
            if (t >= mp.boost_duration) {
                phase = BallisticPhase::PULLUP;
                boost_ended = true;
                pullup_start_time = t;
                // ブースター分離: 弾頭質量に変更
                float dry_mass = mp.initial_mass * (1.0f - mp.fuel_fraction);
                state.mass = dry_mass * mp.warhead_mass_fraction;
                if (verbose) {
                    float spd = std::sqrt(state.vx*state.vx + state.vy*state.vy + state.vz*state.vz);
                    std::cerr << "  HGV BOOST->PULLUP at t=" << t << "s: "
                              << "mass=" << state.mass << "kg, "
                              << "alt=" << (state.z / 1000.0f) << "km, "
                              << "speed=" << spd << " m/s, "
                              << "gamma=" << (gamma * 180.0f / M_PI) << " deg" << std::endl;
                }
            }
        } else if (phase == BallisticPhase::PULLUP) {
            // 物理ベースの遷移条件:
            // 1. 近水平飛行（|gamma| < 5°）かつ十分な時間経過
            // 2. 降下中かつ揚力十分、巡航高度帯（gamma制限緩和）
            // 3. 巡航高度帯で中程度のgamma
            // 4. 拡張タイムアウト
            float rho_here = RHO0 * std::exp(-std::max(state.z, 0.0f) / SCALE_HEIGHT);
            float speed_total = std::sqrt(state.vx*state.vx + state.vy*state.vy + state.vz*state.vz);
            float Cl_here = mp.drag_coefficient * mp.glide_ratio;
            float L_here = 0.5f * rho_here * speed_total * speed_total * Cl_here * mp.cross_section_area;
            float W_here = state.mass * GRAVITY;
            float LW_ratio = L_here / std::max(W_here, 1.0f);

            bool angle_ok = std::abs(gamma) < 0.08f && LW_ratio > 0.5f
                            && (t - pullup_start_time) > 10.0;
            // 巡航高度×1.5以下で降下中、L/W > 0.3なら揚力で引き起こし可能
            bool descending_with_lift = gamma < -0.02f
                                        && LW_ratio > 0.3f
                                        && state.z < mp.cruise_altitude * 1.5f
                                        && state.z > 1000.0f
                                        && (t - pullup_start_time) > 15.0;
            bool near_cruise = std::abs(state.z - mp.cruise_altitude) < 15000.0f
                               && std::abs(gamma) < 0.25f && LW_ratio > 0.1f;
            bool timeout = (t - pullup_start_time) > 300.0 && LW_ratio > 0.05f;

            if (angle_ok || descending_with_lift || near_cruise || timeout) {
                phase = BallisticPhase::GLIDE;
                if (verbose) {
                    float dx_t = xt - state.x;
                    float dy_t = yt - state.y;
                    float dist_t = std::sqrt(dx_t*dx_t + dy_t*dy_t);
                    const char* reason = angle_ok ? "angle_ok" :
                                         descending_with_lift ? "descending_with_lift" :
                                         near_cruise ? "near_cruise" : "timeout";
                    std::cerr << "  HGV PULLUP->GLIDE at t=" << t << "s: "
                              << "alt=" << (state.z / 1000.0f) << "km, "
                              << "gamma=" << (gamma * 180.0f / M_PI) << " deg, "
                              << "L/W=" << LW_ratio << ", "
                              << "dist_to_target=" << (dist_t / 1000.0f) << "km, "
                              << "(" << reason << ")" << std::endl;
                }
            }
        } else if (phase == BallisticPhase::GLIDE) {
            // スキップ回数カウント: vz が正→負に変わった瞬間 = 高度ピーク = 1スキップ完了
            bool vz_positive = state.vz > 0.0f;
            if (prev_vz_positive && !vz_positive) {
                skip_count++;
                if (verbose) {
                    std::cerr << "  HGV Skip #" << skip_count << " at t=" << t << "s: "
                              << "alt=" << (state.z / 1000.0f) << "km" << std::endl;
                }
            }
            prev_vz_positive = vz_positive;

            float dx = xt - state.x;
            float dy = yt - state.y;
            float dist = std::sqrt(dx*dx + dy*dy);

            // TERMINAL遷移: 目標近接 OR 指定スキップ回数に到達
            bool near_target = dist < mp.terminal_dive_range;
            bool skips_done = mp.num_skips > 0 && skip_count >= mp.num_skips;
            if (near_target || skips_done) {
                phase = BallisticPhase::TERMINAL;
                if (verbose) {
                    const char* reason = near_target ? "near target" : "skips done";
                    std::cerr << "  HGV GLIDE->TERMINAL at t=" << t << "s: "
                              << "alt=" << (state.z / 1000.0f) << "km, "
                              << "dist=" << (dist / 1000.0f) << "km"
                              << " (" << reason << ", " << skip_count << " skips)" << std::endl;
                }
            }
        }

        // キャッシュに保存
        TrajectoryPoint tp;
        tp.time = t;
        tp.x = state.x; tp.y = state.y; tp.z = state.z;
        tp.vx = state.vx; tp.vy = state.vy; tp.vz = state.vz;
        tp.mass = state.mass;
        tp.phase = phase;
        tp.object_id = 0;
        params.trajectory_cache.push_back(tp);

        // 目標方位・距離
        float dx = xt - state.x;
        float dy = yt - state.y;
        float distance_to_target = std::sqrt(dx*dx + dy*dy);
        float azimuth_to_target = std::atan2(dy, dx);

        // 分離後は弾頭Cdを使用
        MissileParameters mp_step = mp;
        if (boost_ended) {
            mp_step.drag_coefficient = mp.warhead_cd;
        }

        // RK4ステップ
        state = rk4StepHgv(state, t, dt_physics, mp_step,
                           launch_azimuth, azimuth_to_target, distance_to_target,
                           GRAVITY, EARTH_RADIUS, RHO0, SCALE_HEIGHT, phase);

        // 着弾判定: 高度≤0 のみ（オーバーシュートでも地表到達まで計算を継続）
        bool altitude_impact = boost_ended && state.z <= 0.0f;
        if (altitude_impact) {
            state.z = 0.0f;
            TrajectoryPoint impact;
            impact.time = t + dt_physics;
            impact.x = state.x; impact.y = state.y; impact.z = 0.0f;
            impact.vx = state.vx; impact.vy = state.vy; impact.vz = state.vz;
            impact.mass = state.mass;
            impact.phase = BallisticPhase::TERMINAL;
            impact.object_id = 0;
            params.trajectory_cache.push_back(impact);
            params.death_time = params.birth_time + t + dt_physics;
            break;
        }
    }

    params.trajectory_computed = true;

    if (verbose && !params.trajectory_cache.empty()) {
        float max_alt = 0.0f, max_speed = 0.0f;
        int glide_count = 0, terminal_count = 0;
        for (const auto& tp : params.trajectory_cache) {
            max_alt = std::max(max_alt, tp.z);
            float spd = std::sqrt(tp.vx*tp.vx + tp.vy*tp.vy + tp.vz*tp.vz);
            max_speed = std::max(max_speed, spd);
            if (tp.phase == BallisticPhase::GLIDE) glide_count++;
            if (tp.phase == BallisticPhase::TERMINAL) terminal_count++;
        }
        const auto& last = params.trajectory_cache.back();
        float final_speed = std::sqrt(last.vx*last.vx + last.vy*last.vy + last.vz*last.vz);
        float dx_final = xt - last.x;
        float dy_final = yt - last.y;
        float final_dist = std::sqrt(dx_final*dx_final + dy_final*dy_final);
        float final_heading = std::atan2(last.vy, last.vx);
        float final_azimuth = std::atan2(dy_final, dx_final);
        float hdg_err = final_azimuth - final_heading;
        while (hdg_err > M_PI) hdg_err -= 2.0f * M_PI;
        while (hdg_err < -M_PI) hdg_err += 2.0f * M_PI;

        std::cerr << "  HGV Trajectory computed: "
                  << params.trajectory_cache.size() << " points, "
                  << "flight_time=" << last.time << "s, "
                  << "max_alt=" << (max_alt / 1000.0f) << "km, "
                  << "max_speed=" << max_speed << "m/s ("
                  << (max_speed / 340.0f) << " Mach), "
                  << "skips=" << skip_count << ", "
                  << "glide=" << glide_count << " terminal=" << terminal_count << std::endl;
        std::cerr << "  End state: alt=" << (last.z / 1000.0f) << "km, "
                  << "speed=" << final_speed << "m/s, "
                  << "dist_to_target=" << (final_dist / 1000.0f) << "km, "
                  << "heading_error=" << (hdg_err * 180.0f / M_PI) << " deg" << std::endl;
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

    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    state(0) = tp.x;
    state(1) = tp.y;
    state(2) = tp.z;
    state(3) = tp.vx;
    state(4) = tp.vy;
    state(5) = tp.vz;

    // 加速度は有限差分で計算
    double dt_fd = 0.1;
    auto tp_prev = interpolateCache(params.trajectory_cache, std::max(0.0, elapsed - dt_fd));
    state(6) = (tp.vx - tp_prev.vx) / static_cast<float>(dt_fd);
    state(7) = (tp.vy - tp_prev.vy) / static_cast<float>(dt_fd);
    state(8) = (tp.vz - tp_prev.vz) / static_cast<float>(dt_fd);

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

        // 発射位置（HGVの射程に合わせた距離: 150-350km）
        float launch_range = 150000.0f + uniform_dist_(rng_) * 200000.0f;
        float launch_azimuth = (uniform_dist_(rng_) - 0.5f) * static_cast<float>(M_PI);

        params.initial_state(0) = launch_range * std::cos(launch_azimuth);
        params.initial_state(1) = launch_range * std::sin(launch_azimuth);
        params.initial_state(2) = 0.0f;  // 地上から発射
        params.initial_state(3) = 0.0f;

        // 着弾目標
        float target_range = 5000.0f + uniform_dist_(rng_) * 10000.0f;
        float target_azimuth = (uniform_dist_(rng_) - 0.5f) * 0.3f;
        params.missile_params.target_position(0) = target_range * std::cos(target_azimuth);
        params.missile_params.target_position(1) = target_range * std::sin(target_azimuth);

        // ブーストフェーズ（浅い角度 — depressed trajectory for glide insertion）
        // 0.25-0.40 rad (14-23°): ブースト終了時に巡航高度帯に直接入るのが理想
        params.missile_params.launch_angle = 0.25f + uniform_dist_(rng_) * 0.15f;
        params.missile_params.boost_duration = 40.0f + uniform_dist_(rng_) * 30.0f;  // 40-70s
        params.missile_params.initial_mass = 15000.0f + uniform_dist_(rng_) * 10000.0f;
        params.missile_params.fuel_fraction = 0.65f + uniform_dist_(rng_) * 0.10f;   // 65-75% 燃料
        params.missile_params.specific_impulse = 280.0f + uniform_dist_(rng_) * 50.0f; // Isp 280-330s
        params.missile_params.drag_coefficient = 0.15f + uniform_dist_(rng_) * 0.15f;  // HGV: 低Cd 0.15-0.30
        params.missile_params.cross_section_area = 0.5f + uniform_dist_(rng_) * 0.5f;  // 0.5-1.0 m²

        // HGV固有パラメータ
        params.missile_params.cruise_altitude = 30000.0f + uniform_dist_(rng_) * 20000.0f;
        params.missile_params.glide_ratio = 3.0f + uniform_dist_(rng_) * 3.0f;
        params.missile_params.terminal_dive_range = 15000.0f + uniform_dist_(rng_) * 15000.0f;
        params.missile_params.pullup_duration = 20.0f + uniform_dist_(rng_) * 20.0f;
        params.missile_params.bank_angle_max = 0.7f + uniform_dist_(rng_) * 0.5f;

        // HGVは常に分離（ブースターを投棄）
        params.missile_params.enable_separation = true;
        params.missile_params.warhead_mass_fraction = 0.25f + uniform_dist_(rng_) * 0.15f;
        params.missile_params.warhead_cd = 0.15f + uniform_dist_(rng_) * 0.10f;
        params.missile_params.booster_cd = 1.2f + uniform_dist_(rng_) * 0.6f;

        // death_time は precompute で更新される
        params.death_time = params.birth_time + 1200.0;

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
    StateVector state;
    state.setZero();
    double elapsed = time - params.birth_time;

    if (elapsed < 0) return state;

    // 初回呼び出し時にRK4で事前計算
    if (!params.trajectory_computed) {
        const_cast<TargetGenerator*>(this)->precomputeHgvTrajectory(
            const_cast<TargetParameters&>(params));
    }

    if (params.trajectory_cache.empty()) return state;

    // キャッシュから補間
    auto tp = interpolateCache(params.trajectory_cache, elapsed);

    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    state(0) = tp.x;
    state(1) = tp.y;
    state(2) = tp.z;
    state(3) = tp.vx;
    state(4) = tp.vy;
    state(5) = tp.vz;

    // 加速度は有限差分で計算
    double dt_fd = 0.1;
    auto tp_prev = interpolateCache(params.trajectory_cache, std::max(0.0, elapsed - dt_fd));
    state(6) = (tp.vx - tp_prev.vx) / static_cast<float>(dt_fd);
    state(7) = (tp.vy - tp_prev.vy) / static_cast<float>(dt_fd);
    state(8) = (tp.vz - tp_prev.vz) / static_cast<float>(dt_fd);

    return state;
}

} // namespace fasttracker
