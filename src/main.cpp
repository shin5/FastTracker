#pragma execution_character_set("utf-8")

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include "tracker/multi_target_tracker.hpp"
#include "simulation/target_generator.hpp"
#include "simulation/radar_simulator.hpp"
#include "evaluation/tracking_evaluator.hpp"
#include "utils/cuda_utils.cuh"

using namespace fasttracker;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// meters → lat/lon 変換（基準点からの相対座標）
static void metersToLatLon(float x_m, float y_m,
                           double ref_lat, double ref_lon,
                           double& out_lat, double& out_lon) {
    const double R = 6371000.0;
    double ref_lat_r = ref_lat * M_PI / 180.0;
    double dlat = y_m / R;
    double dlon = x_m / (R * std::cos(ref_lat_r));
    out_lat = ref_lat + dlat * 180.0 / M_PI;
    out_lon = ref_lon + dlon * 180.0 / M_PI;
}

static const char* phaseToString(BallisticPhase phase) {
    switch (phase) {
        case BallisticPhase::BOOST: return "BOOST";
        case BallisticPhase::MIDCOURSE: return "MIDCOURSE";
        case BallisticPhase::TERMINAL: return "TERMINAL";
        default: return "UNKNOWN";
    }
}

// ========================================
// 自動パラメータ調整ヘルパー
// ========================================

struct AutoAdjustResult {
    float launch_angle;
    float specific_impulse;
    float fuel_fraction;
    float impact_distance;  // meters
    int iterations;
};

// 着弾距離計算（軽量版 - 着弾点のみ取得）
static float computeImpactDistance(
    TargetGenerator& gen,
    float launch_x, float launch_y,
    float target_x, float target_y,
    float launch_angle, float boost_duration,
    float initial_mass, float fuel_fraction,
    float specific_impulse, float drag_coefficient,
    float cross_section_area)
{
    TargetParameters missile;
    missile.motion_model = MotionModel::BALLISTIC_MISSILE;
    missile.birth_time = 0.0;
    missile.death_time = 3600.0;

    missile.initial_state.setZero();
    missile.initial_state(0) = launch_x;
    missile.initial_state(1) = launch_y;

    missile.missile_params.launch_angle = launch_angle;
    missile.missile_params.boost_duration = boost_duration;
    missile.missile_params.initial_mass = initial_mass;
    missile.missile_params.fuel_fraction = fuel_fraction;
    missile.missile_params.specific_impulse = specific_impulse;
    missile.missile_params.drag_coefficient = drag_coefficient;
    missile.missile_params.cross_section_area = cross_section_area;
    missile.missile_params.target_position(0) = target_x;
    missile.missile_params.target_position(1) = target_y;

    gen.precomputeBallisticTrajectory(missile, false);

    if (missile.trajectory_cache.empty()) return 1e12f;

    const auto& last = missile.trajectory_cache.back();
    float dx = last.x - target_x;
    float dy = last.y - target_y;
    return std::sqrt(dx * dx + dy * dy);
}

// 発射角の最適化（粗→精密の2段階探索）
static float optimizeLaunchAngle(
    TargetGenerator& gen,
    float launch_x, float launch_y,
    float target_x, float target_y,
    float boost_duration,
    float initial_mass, float fuel_fraction,
    float specific_impulse, float drag_coefficient,
    float cross_section_area,
    float& out_best_dist,
    int& iteration_count)
{
    float best_angle = 0.7f;
    float best_dist = 1e12f;

    // 粗探索: 0.25〜1.30 rad, step=0.05 (22点)
    for (float angle = 0.25f; angle <= 1.301f; angle += 0.05f) {
        float dist = computeImpactDistance(gen, launch_x, launch_y, target_x, target_y,
            angle, boost_duration, initial_mass, fuel_fraction,
            specific_impulse, drag_coefficient, cross_section_area);
        iteration_count++;
        if (dist < best_dist) {
            best_dist = dist;
            best_angle = angle;
        }
    }

    // 精密探索: 最良値±0.05 rad, step=0.003 (34点)
    float lo = std::max(0.25f, best_angle - 0.05f);
    float hi = std::min(1.30f, best_angle + 0.05f);
    for (float angle = lo; angle <= hi + 0.001f; angle += 0.003f) {
        float dist = computeImpactDistance(gen, launch_x, launch_y, target_x, target_y,
            angle, boost_duration, initial_mass, fuel_fraction,
            specific_impulse, drag_coefficient, cross_section_area);
        iteration_count++;
        if (dist < best_dist) {
            best_dist = dist;
            best_angle = angle;
        }
    }

    out_best_dist = best_dist;
    return best_angle;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --mode <mode>        Mode: tracker (default), trajectory, auto-adjust" << std::endl;
    std::cout << "  --num-targets <N>    Number of targets (default: 1000)" << std::endl;
    std::cout << "  --duration <T>       Simulation duration in seconds (default: 10.0)" << std::endl;
    std::cout << "  --framerate <FPS>    Frame rate in Hz (default: 30)" << std::endl;
    std::cout << "  --output <file>      Output file for results (default: results.csv)" << std::endl;
    std::cout << "  --scenario <type>    Scenario type:" << std::endl;
    std::cout << "                         default, clustered, high-maneuver, ballistic," << std::endl;
    std::cout << "                         hypersonic, mixed-threat, single-ballistic" << std::endl;
    std::cout << "  --launch-x/y <m>    Launch position in meters" << std::endl;
    std::cout << "  --target-x/y <m>    Target position in meters" << std::endl;
    std::cout << "  --target-lat/lon <deg> Target lat/lon for coordinate conversion" << std::endl;
    std::cout << "  --launch-angle <rad> Launch angle in radians" << std::endl;
    std::cout << "  --boost-duration <s> Boost phase duration" << std::endl;
    std::cout << "  --boost-accel <m/s2> Boost acceleration (legacy)" << std::endl;
    std::cout << "  --initial-mass <kg>  Missile initial mass (default: 20000)" << std::endl;
    std::cout << "  --fuel-fraction <f>  Fuel mass fraction (default: 0.65)" << std::endl;
    std::cout << "  --specific-impulse <s> Specific impulse (default: 250)" << std::endl;
    std::cout << "  --drag-coefficient <cd> Drag coefficient (default: 0.3)" << std::endl;
    std::cout << "  --cross-section <m2> Cross section area (default: 1.0)" << std::endl;
    std::cout << "  --sensor-x/y <m>    Sensor position in meters" << std::endl;
    std::cout << "  --radar-max-range <m> Radar max range (0=auto)" << std::endl;
    std::cout << "  --radar-fov <rad>    Radar field of view" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
}

// ========================================
// trajectory mode: 軌道CSVのみ出力
// ========================================
static int runTrajectoryMode(
    float launch_x, float launch_y,
    float target_x, float target_y,
    float launch_angle, float boost_duration,
    float initial_mass, float fuel_fraction,
    float specific_impulse, float drag_coefficient,
    float cross_section_area,
    double target_lat, double target_lon,
    double dt_output)
{
    // TargetParameters設定
    TargetParameters missile;
    missile.motion_model = MotionModel::BALLISTIC_MISSILE;
    missile.birth_time = 0.0;
    missile.death_time = 3600.0;  // precomputeで更新される

    missile.initial_state(0) = launch_x;
    missile.initial_state(1) = launch_y;
    missile.initial_state(2) = 0.0f;
    missile.initial_state(3) = 0.0f;

    missile.missile_params.launch_angle = launch_angle;
    missile.missile_params.boost_duration = boost_duration;
    missile.missile_params.initial_mass = initial_mass;
    missile.missile_params.fuel_fraction = fuel_fraction;
    missile.missile_params.specific_impulse = specific_impulse;
    missile.missile_params.drag_coefficient = drag_coefficient;
    missile.missile_params.cross_section_area = cross_section_area;
    missile.missile_params.target_position(0) = target_x;
    missile.missile_params.target_position(1) = target_y;

    // RK4で軌道計算
    TargetGenerator gen(1);
    gen.precomputeBallisticTrajectory(missile);

    if (missile.trajectory_cache.empty()) {
        std::cerr << "ERROR: Trajectory computation failed" << std::endl;
        return 1;
    }

    // trajectory.csv 出力
    std::ofstream traj_file("trajectory.csv");
    traj_file << "time,x,y,altitude,vx,vy,vz,speed,phase,lat,lon" << std::endl;
    traj_file << std::fixed;

    float max_alt = 0.0f;
    float max_speed = 0.0f;
    float flight_duration = static_cast<float>(missile.trajectory_cache.back().time);
    int num_output = 0;

    // dt_output間隔で出力（キャッシュは0.05s間隔、出力はユーザー指定間隔）
    for (double t = 0.0; t <= flight_duration + 0.001; t += dt_output) {
        // キャッシュから補間
        const auto& cache = missile.trajectory_cache;
        TrajectoryPoint tp;

        if (t <= cache.front().time) {
            tp = cache.front();
        } else if (t >= cache.back().time) {
            tp = cache.back();
        } else {
            int lo = 0, hi = static_cast<int>(cache.size()) - 1;
            while (lo < hi - 1) {
                int mid = (lo + hi) / 2;
                if (cache[mid].time <= t) lo = mid; else hi = mid;
            }
            float frac = static_cast<float>((t - cache[lo].time) / (cache[hi].time - cache[lo].time));
            tp.time = t;
            tp.x = cache[lo].x + frac * (cache[hi].x - cache[lo].x);
            tp.y = cache[lo].y + frac * (cache[hi].y - cache[lo].y);
            tp.z = cache[lo].z + frac * (cache[hi].z - cache[lo].z);
            tp.vx = cache[lo].vx + frac * (cache[hi].vx - cache[lo].vx);
            tp.vy = cache[lo].vy + frac * (cache[hi].vy - cache[lo].vy);
            tp.vz = cache[lo].vz + frac * (cache[hi].vz - cache[lo].vz);
            tp.phase = (frac < 0.5f) ? cache[lo].phase : cache[hi].phase;
        }

        float speed = std::sqrt(tp.vx*tp.vx + tp.vy*tp.vy + tp.vz*tp.vz);
        max_alt = std::max(max_alt, tp.z);
        max_speed = std::max(max_speed, speed);

        // lat/lon変換
        double lat = 0.0, lon = 0.0;
        metersToLatLon(tp.x, tp.y, target_lat, target_lon, lat, lon);

        traj_file << std::setprecision(4) << tp.time << ","
                  << std::setprecision(2) << tp.x << ","
                  << tp.y << ","
                  << tp.z << ","
                  << tp.vx << ","
                  << tp.vy << ","
                  << tp.vz << ","
                  << std::setprecision(1) << speed << ","
                  << phaseToString(tp.phase) << ","
                  << std::setprecision(6) << lat << ","
                  << lon << std::endl;

        num_output++;
    }

    traj_file.close();

    float range_m = std::sqrt((launch_x - target_x) * (launch_x - target_x) +
                              (launch_y - target_y) * (launch_y - target_y));

    // サマリーをstdoutに出力（FlaskがパースするJSON形式）
    std::cout << std::fixed;
    std::cout << "{" << std::endl;
    std::cout << "  \"range_km\": " << std::setprecision(1) << (range_m / 1000.0f) << "," << std::endl;
    std::cout << "  \"flight_duration\": " << std::setprecision(1) << flight_duration << "," << std::endl;
    std::cout << "  \"max_altitude\": " << std::setprecision(1) << max_alt << "," << std::endl;
    std::cout << "  \"max_speed\": " << std::setprecision(1) << max_speed << "," << std::endl;
    std::cout << "  \"num_steps\": " << num_output << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}

// ========================================
// auto-adjust mode: パラメータ自動調整 + 軌道出力
// ========================================
static int runAutoAdjustMode(
    float launch_x, float launch_y,
    float target_x, float target_y,
    float launch_angle_orig, float boost_duration,
    float initial_mass, float fuel_fraction_orig,
    float specific_impulse_orig, float drag_coefficient,
    float cross_section_area,
    double target_lat, double target_lon,
    double dt_output,
    float distance_threshold_m)
{
    TargetGenerator gen(1);
    int iterations = 0;

    // 現実的なパラメータ上限
    const float MAX_ISP = 350.0f;           // 液体燃料ロケット上限
    const float MAX_FUEL_FRACTION = 0.85f;  // 構造限界
    const float ISP_STEP = 10.0f;
    const float FF_STEP = 0.02f;

    float best_angle = launch_angle_orig;
    float best_isp = specific_impulse_orig;
    float best_ff = fuel_fraction_orig;
    float best_dist = 1e12f;

    // Phase 1: 現在のパラメータで発射角のみ最適化
    best_angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
        boost_duration, initial_mass, best_ff, best_isp,
        drag_coefficient, cross_section_area, best_dist, iterations);

    std::cerr << "Phase 1 (angle only): angle=" << best_angle
              << " rad, dist=" << (best_dist / 1000.0f) << " km, iters=" << iterations << std::endl;

    // Phase 2: Isp を段階的に増加
    if (best_dist > distance_threshold_m && best_isp < MAX_ISP) {
        for (float isp = specific_impulse_orig + ISP_STEP; isp <= MAX_ISP + 0.1f; isp += ISP_STEP) {
            float dist = 0.0f;
            float angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
                boost_duration, initial_mass, best_ff, isp,
                drag_coefficient, cross_section_area, dist, iterations);
            if (dist < best_dist) {
                best_dist = dist;
                best_angle = angle;
                best_isp = isp;
            }
            if (best_dist <= distance_threshold_m) break;
        }
        std::cerr << "Phase 2 (Isp adj): isp=" << best_isp
                  << ", angle=" << best_angle
                  << ", dist=" << (best_dist / 1000.0f) << " km" << std::endl;
    }

    // Phase 3: 燃料割合を段階的に増加（Ispも各段で最適化）
    if (best_dist > distance_threshold_m && best_ff < MAX_FUEL_FRACTION) {
        for (float ff = fuel_fraction_orig + FF_STEP; ff <= MAX_FUEL_FRACTION + 0.001f; ff += FF_STEP) {
            // この燃料割合で、全Isp範囲を試す
            for (float isp = specific_impulse_orig; isp <= MAX_ISP + 0.1f; isp += ISP_STEP) {
                float dist = 0.0f;
                float angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
                    boost_duration, initial_mass, ff, isp,
                    drag_coefficient, cross_section_area, dist, iterations);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_angle = angle;
                    best_isp = isp;
                    best_ff = ff;
                }
            }
            if (best_dist <= distance_threshold_m) break;
        }
        std::cerr << "Phase 3 (fuel adj): ff=" << best_ff
                  << ", isp=" << best_isp
                  << ", angle=" << best_angle
                  << ", dist=" << (best_dist / 1000.0f) << " km" << std::endl;
    }

    std::cerr << "Auto-adjust complete: " << iterations << " iterations, "
              << "impact distance=" << (best_dist / 1000.0f) << " km" << std::endl;

    // 最適パラメータで軌道を生成
    TargetParameters missile;
    missile.motion_model = MotionModel::BALLISTIC_MISSILE;
    missile.birth_time = 0.0;
    missile.death_time = 3600.0;

    missile.initial_state.setZero();
    missile.initial_state(0) = launch_x;
    missile.initial_state(1) = launch_y;

    missile.missile_params.launch_angle = best_angle;
    missile.missile_params.boost_duration = boost_duration;
    missile.missile_params.initial_mass = initial_mass;
    missile.missile_params.fuel_fraction = best_ff;
    missile.missile_params.specific_impulse = best_isp;
    missile.missile_params.drag_coefficient = drag_coefficient;
    missile.missile_params.cross_section_area = cross_section_area;
    missile.missile_params.target_position(0) = target_x;
    missile.missile_params.target_position(1) = target_y;

    gen.precomputeBallisticTrajectory(missile, true);

    if (missile.trajectory_cache.empty()) {
        std::cerr << "ERROR: Final trajectory computation failed" << std::endl;
        return 1;
    }

    // trajectory.csv 出力（runTrajectoryModeと同じフォーマット）
    std::ofstream traj_file("trajectory.csv");
    traj_file << "time,x,y,altitude,vx,vy,vz,speed,phase,lat,lon" << std::endl;
    traj_file << std::fixed;

    float max_alt = 0.0f;
    float max_speed = 0.0f;
    float flight_duration = static_cast<float>(missile.trajectory_cache.back().time);
    int num_output = 0;

    for (double t = 0.0; t <= flight_duration + 0.001; t += dt_output) {
        const auto& cache = missile.trajectory_cache;
        TrajectoryPoint tp;

        if (t <= cache.front().time) {
            tp = cache.front();
        } else if (t >= cache.back().time) {
            tp = cache.back();
        } else {
            int lo = 0, hi = static_cast<int>(cache.size()) - 1;
            while (lo < hi - 1) {
                int mid = (lo + hi) / 2;
                if (cache[mid].time <= t) lo = mid; else hi = mid;
            }
            float frac = static_cast<float>((t - cache[lo].time) / (cache[hi].time - cache[lo].time));
            tp.time = t;
            tp.x = cache[lo].x + frac * (cache[hi].x - cache[lo].x);
            tp.y = cache[lo].y + frac * (cache[hi].y - cache[lo].y);
            tp.z = cache[lo].z + frac * (cache[hi].z - cache[lo].z);
            tp.vx = cache[lo].vx + frac * (cache[hi].vx - cache[lo].vx);
            tp.vy = cache[lo].vy + frac * (cache[hi].vy - cache[lo].vy);
            tp.vz = cache[lo].vz + frac * (cache[hi].vz - cache[lo].vz);
            tp.phase = (frac < 0.5f) ? cache[lo].phase : cache[hi].phase;
        }

        float speed = std::sqrt(tp.vx*tp.vx + tp.vy*tp.vy + tp.vz*tp.vz);
        max_alt = std::max(max_alt, tp.z);
        max_speed = std::max(max_speed, speed);

        double lat = 0.0, lon = 0.0;
        metersToLatLon(tp.x, tp.y, target_lat, target_lon, lat, lon);

        traj_file << std::setprecision(4) << tp.time << ","
                  << std::setprecision(2) << tp.x << ","
                  << tp.y << ","
                  << tp.z << ","
                  << tp.vx << ","
                  << tp.vy << ","
                  << tp.vz << ","
                  << std::setprecision(1) << speed << ","
                  << phaseToString(tp.phase) << ","
                  << std::setprecision(6) << lat << ","
                  << lon << std::endl;

        num_output++;
    }

    traj_file.close();

    float range_m = std::sqrt((launch_x - target_x) * (launch_x - target_x) +
                              (launch_y - target_y) * (launch_y - target_y));

    // JSON出力（調整結果付き）
    bool was_adjusted = (best_angle != launch_angle_orig ||
                         best_isp != specific_impulse_orig ||
                         best_ff != fuel_fraction_orig);

    std::cout << std::fixed;
    std::cout << "{" << std::endl;
    std::cout << "  \"range_km\": " << std::setprecision(1) << (range_m / 1000.0f) << "," << std::endl;
    std::cout << "  \"flight_duration\": " << std::setprecision(1) << flight_duration << "," << std::endl;
    std::cout << "  \"max_altitude\": " << std::setprecision(1) << max_alt << "," << std::endl;
    std::cout << "  \"max_speed\": " << std::setprecision(1) << max_speed << "," << std::endl;
    std::cout << "  \"num_steps\": " << num_output << "," << std::endl;
    std::cout << "  \"adjusted\": " << (was_adjusted ? "true" : "false") << "," << std::endl;
    std::cout << "  \"adj_launch_angle\": " << std::setprecision(4) << best_angle << "," << std::endl;
    std::cout << "  \"adj_specific_impulse\": " << std::setprecision(1) << best_isp << "," << std::endl;
    std::cout << "  \"adj_fuel_fraction\": " << std::setprecision(3) << best_ff << "," << std::endl;
    std::cout << "  \"impact_distance_km\": " << std::setprecision(1) << (best_dist / 1000.0f) << "," << std::endl;
    std::cout << "  \"search_iterations\": " << iterations << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}

// ========================================
// tracker mode: 従来のトラッカー実行
// ========================================
static int runTrackerMode(
    int num_targets, double duration, double framerate,
    const std::string& output_file, const std::string& scenario,
    float launch_x, float launch_y, float target_x, float target_y,
    float launch_angle, float boost_duration, float boost_accel,
    float initial_mass, float fuel_fraction,
    float specific_impulse, float drag_coefficient,
    float cross_section_area,
    float sensor_x, float sensor_y,
    float radar_max_range, float radar_fov)
{
    // CUDA デバイス情報
    try {
        auto device_info = cuda::DeviceInfo::getCurrent();
        device_info.print();
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
        return 1;
    }

    // シミュレーション環境のセットアップ
    TargetGenerator target_gen(num_targets);

    if (scenario == "default") {
        target_gen.initializeDefaultScenario();
    } else if (scenario == "clustered") {
        target_gen.generateClusteredScenario(Eigen::Vector2f(0.0f, 0.0f), 500.0f);
    } else if (scenario == "high-maneuver") {
        target_gen.generateHighManeuverScenario();
    } else if (scenario == "ballistic") {
        std::cout << "Ballistic Missile Scenario" << std::endl;
        target_gen.generateBallisticMissileScenario(num_targets);
    } else if (scenario == "hypersonic") {
        std::cout << "Hypersonic Glide Vehicle Scenario" << std::endl;
        target_gen.generateHypersonicGlideScenario(num_targets);
    } else if (scenario == "mixed-threat") {
        std::cout << "Mixed Threat Scenario (Aircraft + Missiles)" << std::endl;
        target_gen.generateMixedThreatScenario();
    } else if (scenario == "single-ballistic") {
        float range_m = std::sqrt((launch_x - target_x) * (launch_x - target_x) +
                                  (launch_y - target_y) * (launch_y - target_y));
        std::cout << "Single Ballistic Missile Scenario (" << (range_m / 1000.0f) << "km range)" << std::endl;

        TargetParameters missile;
        missile.motion_model = MotionModel::BALLISTIC_MISSILE;
        missile.birth_time = 0.0;
        missile.death_time = duration;  // precomputeで更新される

        missile.initial_state(0) = launch_x;
        missile.initial_state(1) = launch_y;
        missile.initial_state(2) = 0.0f;
        missile.initial_state(3) = 0.0f;

        missile.missile_params.launch_angle = launch_angle;
        missile.missile_params.max_altitude = 120000.0f;
        missile.missile_params.boost_duration = boost_duration;
        missile.missile_params.boost_acceleration = boost_accel;
        missile.missile_params.initial_mass = initial_mass;
        missile.missile_params.fuel_fraction = fuel_fraction;
        missile.missile_params.specific_impulse = specific_impulse;
        missile.missile_params.drag_coefficient = drag_coefficient;
        missile.missile_params.cross_section_area = cross_section_area;
        missile.missile_params.target_position(0) = target_x;
        missile.missile_params.target_position(1) = target_y;

        std::vector<TargetParameters> params = {missile};
        target_gen.initializeCustomScenario(params);

        // 軌道を事前計算して実際のflight_durationを取得
        auto& target_params = const_cast<std::vector<TargetParameters>&>(target_gen.getTargetParams());
        target_gen.precomputeBallisticTrajectory(target_params[0]);

        // durationを実際の飛翔時間に更新
        if (target_params[0].trajectory_computed) {
            duration = target_params[0].death_time;
            std::cout << "  Flight duration (physics): " << duration << "s" << std::endl;
        }

        std::cout << "  Launch: (" << (launch_x / 1000.0f) << "km, " << (launch_y / 1000.0f) << "km)" << std::endl;
        std::cout << "  Target: (" << (target_x / 1000.0f) << "km, " << (target_y / 1000.0f) << "km)" << std::endl;
        std::cout << "  Range: " << (range_m / 1000.0f) << " km" << std::endl;
        std::cout << "  Boost: " << boost_duration << "s" << std::endl;
        std::cout << "  Launch angle: " << launch_angle << " rad" << std::endl;
        std::cout << "  Mass: " << initial_mass << "kg, fuel=" << (fuel_fraction * 100) << "%" << std::endl;
        std::cout << "  Isp: " << specific_impulse << "s, Cd=" << drag_coefficient << std::endl;
    } else {
        std::cerr << "Unknown scenario: " << scenario << std::endl;
        return 1;
    }

    // レーダーパラメータ
    RadarParameters radar_params;
    if (scenario == "single-ballistic") {
        radar_params.sensor_x = sensor_x;
        radar_params.sensor_y = sensor_y;

        float dx_sensor_launch = launch_x - sensor_x;
        float dy_sensor_launch = launch_y - sensor_y;
        float sensor_to_launch = std::sqrt(dx_sensor_launch * dx_sensor_launch + dy_sensor_launch * dy_sensor_launch);
        float dx_sensor_target = target_x - sensor_x;
        float dy_sensor_target = target_y - sensor_y;
        float sensor_to_target = std::sqrt(dx_sensor_target * dx_sensor_target + dy_sensor_target * dy_sensor_target);
        float max_sensor_dist = std::max(sensor_to_launch, sensor_to_target);

        if (radar_max_range > 0.0f) {
            radar_params.max_range = radar_max_range;
        } else {
            radar_params.max_range = max_sensor_dist * 1.3f;
        }

        radar_params.field_of_view = radar_fov;

        float snr_at_max = 20.0f;
        radar_params.snr_ref = snr_at_max - 40.0f * std::log10(1000.0f / radar_params.max_range);
        radar_params.false_alarm_rate = 0.1f / (M_PI * radar_params.max_range * radar_params.max_range);

        std::cout << "  Sensor position: (" << (sensor_x / 1000.0f) << "km, " << (sensor_y / 1000.0f) << "km)" << std::endl;
        std::cout << "  Radar max_range: " << (radar_params.max_range / 1000.0f) << " km" << std::endl;
    } else if (scenario == "ballistic" || scenario == "hypersonic" || scenario == "mixed-threat") {
        radar_params.max_range = 150000.0f;
        radar_params.false_alarm_rate = 1e-10f;
        radar_params.snr_ref = 110.0f;
    }

    RadarSimulator radar_sim(target_gen, radar_params);

    // トラッカーパラメータ
    ProcessNoise process_noise;
    AssociationParams assoc_params;

    if (scenario == "single-ballistic") {
        process_noise.position_noise = 50.0f;
        process_noise.velocity_noise = 20.0f;
        process_noise.accel_noise = 50.0f;

        assoc_params.gate_threshold = 200.0f;
        assoc_params.max_distance = 100.0f;
        assoc_params.confirm_hits = 3;
        assoc_params.confirm_window = 5;
        assoc_params.delete_misses = 10;
        assoc_params.min_snr_for_init = 22.0f;
    } else if (scenario == "ballistic" || scenario == "hypersonic" || scenario == "mixed-threat") {
        process_noise.position_noise = 4.0f;
        process_noise.velocity_noise = 8.0f;
        process_noise.accel_noise = 4.0f;

        assoc_params.gate_threshold = 15.0f;
        assoc_params.max_distance = 4.0f;
        assoc_params.confirm_hits = 3;
        assoc_params.confirm_window = 4;
        assoc_params.delete_misses = 8;
        assoc_params.min_snr_for_init = 40.0f;
    }

    int max_tracks = std::max(num_targets * 2, 10);
    UKFParams ukf_params;
    if (scenario == "single-ballistic") {
        ukf_params.alpha = 0.1f;
        ukf_params.lambda = ukf_params.alpha * ukf_params.alpha *
                            (STATE_DIM + ukf_params.kappa) - STATE_DIM;
    }
    MultiTargetTracker tracker(max_tracks, ukf_params, assoc_params, process_noise);
    tracker.setSensorPosition(sensor_x, sensor_y);

    float ospa_cutoff = (scenario == "single-ballistic") ? 10000.0f : 100.0f;
    TrackingEvaluator evaluator(ospa_cutoff, 2);

    // シミュレーションループ
    double dt = 1.0 / framerate;
    int num_frames = static_cast<int>(duration / dt);

    std::cout << "\nStarting simulation..." << std::endl;
    std::cout << "Total frames: " << num_frames << std::endl;

    std::ofstream out_file(output_file);
    out_file << "frame,time,num_tracks,num_confirmed,num_measurements,processing_time_ms" << std::endl;

    std::ofstream track_file("track_details.csv");
    track_file << "frame,time,track_id,x,y,vx,vy,ax,ay,state,model_prob_cv,model_prob_high,model_prob_med" << std::endl;

    std::ofstream ground_truth_file("ground_truth.csv");
    ground_truth_file << "frame,time,target_id,x,y,vx,vy,ax,ay,altitude" << std::endl;

    std::ofstream meas_file("measurements.csv");
    meas_file << "frame,time,range,azimuth,elevation,doppler,snr" << std::endl;

    for (int frame = 0; frame < num_frames; frame++) {
        double current_time = frame * dt;

        auto ground_truth = radar_sim.getTrueStates(current_time);

        for (size_t i = 0; i < ground_truth.size(); i++) {
            const auto& gt = ground_truth[i];
            float altitude = target_gen.getLastAltitude(static_cast<int>(i));
            ground_truth_file << frame << ","
                            << current_time << ","
                            << i << ","
                            << gt(0) << "," << gt(1) << ","
                            << gt(2) << "," << gt(3) << ","
                            << gt(4) << "," << gt(5) << ","
                            << altitude << std::endl;
        }

        auto measurements = radar_sim.generate(current_time);

        for (const auto& m : measurements) {
            meas_file << frame << ","
                      << current_time << ","
                      << m.range << "," << m.azimuth << ","
                      << m.elevation << "," << m.doppler << ","
                      << m.snr << std::endl;
        }

        try {
            tracker.update(measurements, current_time);
        } catch (const std::exception& e) {
            std::cerr << "Exception in tracker.update() at frame " << frame
                      << ": " << e.what() << std::endl;
            return 1;
        }

        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cerr << "CUDA Error after frame " << frame << ": "
                      << cudaGetErrorString(cuda_err) << std::endl;
            return 1;
        }

        auto tracks = tracker.getAllTracks();
        evaluator.update(tracks, ground_truth,
                        static_cast<int>(measurements.size()),
                        current_time);

        for (const auto& track : tracks) {
            int state_value = 0;
            if (track.track_state == TrackState::CONFIRMED) state_value = 1;
            else if (track.track_state == TrackState::LOST) state_value = 2;

            float prob_cv = track.model_probs.size() >= 1 ? track.model_probs[0] : 0.333f;
            float prob_high = track.model_probs.size() >= 2 ? track.model_probs[1] : 0.333f;
            float prob_med = track.model_probs.size() >= 3 ? track.model_probs[2] : 0.333f;

            track_file << frame << "," << current_time << ","
                      << track.id << ","
                      << track.state(0) << "," << track.state(1) << ","
                      << track.state(2) << "," << track.state(3) << ","
                      << track.state(4) << "," << track.state(5) << ","
                      << state_value << ","
                      << prob_cv << "," << prob_high << "," << prob_med << std::endl;
        }

        const auto& perf = tracker.getLastPerformanceStats();

        out_file << frame << "," << current_time << ","
                 << tracker.getNumTracks() << ","
                 << tracker.getNumConfirmedTracks() << ","
                 << measurements.size() << ","
                 << perf.total_time_ms << std::endl;

        if ((frame + 1) % 10 == 0 || frame == 0) {
            std::cout << "Frame " << (frame + 1) << "/" << num_frames
                      << " | Tracks: " << tracker.getNumConfirmedTracks()
                      << " | Meas: " << measurements.size()
                      << " | Time: " << std::fixed << std::setprecision(2)
                      << perf.total_time_ms << " ms" << std::endl;
        }
    }

    out_file.close();
    track_file.close();
    ground_truth_file.close();
    meas_file.close();

    std::cout << "\n=== Simulation Complete ===" << std::endl;

    tracker.printStatistics();
    radar_sim.printStatistics();
    evaluator.printSummary();
    evaluator.exportToCSV("evaluation_results.csv");

    return 0;
}

// ========================================
// main
// ========================================
int main(int argc, char** argv) {
    // デフォルトパラメータ
    std::string mode = "tracker";
    int num_targets = 1000;
    double duration = 10.0;
    double framerate = 30.0;
    std::string output_file = "results.csv";
    std::string scenario = "default";

    // ミサイルパラメータ
    float launch_x = 353553.0f, launch_y = 353553.0f;
    float target_x = 0.0f, target_y = 0.0f;
    float launch_angle = 0.7f;
    float boost_duration = 65.0f;
    float boost_accel = 30.0f;

    // 物理パラメータ
    float initial_mass = 20000.0f;
    float fuel_fraction = 0.65f;
    float specific_impulse = 250.0f;
    float drag_coefficient = 0.3f;
    float cross_section_area = 1.0f;

    // センサーパラメータ
    float sensor_x = 0.0f, sensor_y = 0.0f;
    float radar_max_range = 0.0f;
    float radar_fov = 2.0f * static_cast<float>(M_PI);

    // 座標変換用
    double target_lat = 35.6762, target_lon = 139.6503;

    // 自動調整用
    float distance_threshold = 10000.0f;  // 10km デフォルト

    // コマンドライン引数解析
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--num-targets" && i + 1 < argc) num_targets = std::atoi(argv[++i]);
        else if (arg == "--duration" && i + 1 < argc) duration = std::atof(argv[++i]);
        else if (arg == "--framerate" && i + 1 < argc) framerate = std::atof(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--scenario" && i + 1 < argc) scenario = argv[++i];
        else if (arg == "--launch-x" && i + 1 < argc) launch_x = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--launch-y" && i + 1 < argc) launch_y = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--target-x" && i + 1 < argc) target_x = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--target-y" && i + 1 < argc) target_y = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--target-lat" && i + 1 < argc) target_lat = std::atof(argv[++i]);
        else if (arg == "--target-lon" && i + 1 < argc) target_lon = std::atof(argv[++i]);
        else if (arg == "--launch-angle" && i + 1 < argc) launch_angle = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--boost-duration" && i + 1 < argc) boost_duration = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--boost-accel" && i + 1 < argc) boost_accel = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--initial-mass" && i + 1 < argc) initial_mass = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--fuel-fraction" && i + 1 < argc) fuel_fraction = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--specific-impulse" && i + 1 < argc) specific_impulse = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--drag-coefficient" && i + 1 < argc) drag_coefficient = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--cross-section" && i + 1 < argc) cross_section_area = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--sensor-x" && i + 1 < argc) sensor_x = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--sensor-y" && i + 1 < argc) sensor_y = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--radar-max-range" && i + 1 < argc) radar_max_range = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--radar-fov" && i + 1 < argc) radar_fov = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--distance-threshold" && i + 1 < argc) distance_threshold = static_cast<float>(std::atof(argv[++i]));
    }

    if (mode == "trajectory") {
        return runTrajectoryMode(
            launch_x, launch_y, target_x, target_y,
            launch_angle, boost_duration,
            initial_mass, fuel_fraction, specific_impulse,
            drag_coefficient, cross_section_area,
            target_lat, target_lon,
            1.0 / framerate);
    }

    if (mode == "auto-adjust") {
        return runAutoAdjustMode(
            launch_x, launch_y, target_x, target_y,
            launch_angle, boost_duration,
            initial_mass, fuel_fraction, specific_impulse,
            drag_coefficient, cross_section_area,
            target_lat, target_lon,
            1.0 / framerate,
            distance_threshold);
    }

    std::cout << "=== FastTracker: GPU-Accelerated Multi-Target Tracker ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Targets: " << num_targets << std::endl;
    std::cout << "  Duration: " << duration << " s" << std::endl;
    std::cout << "  Frame rate: " << framerate << " Hz" << std::endl;
    std::cout << "  Scenario: " << scenario << std::endl;
    std::cout << "==========================================================" << std::endl;

    return runTrackerMode(
        num_targets, duration, framerate, output_file, scenario,
        launch_x, launch_y, target_x, target_y,
        launch_angle, boost_duration, boost_accel,
        initial_mass, fuel_fraction, specific_impulse,
        drag_coefficient, cross_section_area,
        sensor_x, sensor_y, radar_max_range, radar_fov);
}
