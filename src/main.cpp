#pragma execution_character_set("utf-8")

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_map>
#include <unordered_set>
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
        case BallisticPhase::PULLUP: return "PULLUP";
        case BallisticPhase::GLIDE: return "GLIDE";
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

// 着弾距離＋最高高度計算
struct TrajectoryResult {
    float impact_distance;
    float max_altitude;
};

// HGV固有パラメータ（auto-adjust用）
struct HgvParams {
    float cruise_altitude = 40000.0f;
    float glide_ratio = 4.0f;
    float terminal_dive_range = 20000.0f;
    float pullup_duration = 30.0f;
    float bank_angle_max = 0.0f;
    int num_skips = 1;
};

static TrajectoryResult computeTrajectoryResult(
    TargetGenerator& gen,
    float launch_x, float launch_y,
    float target_x, float target_y,
    float launch_angle, float boost_duration,
    float initial_mass, float fuel_fraction,
    float specific_impulse, float drag_coefficient,
    float cross_section_area,
    bool is_hgv = false,
    const HgvParams& hgv = HgvParams())
{
    TargetParameters missile;
    missile.motion_model = is_hgv ? MotionModel::HYPERSONIC_GLIDE : MotionModel::BALLISTIC_MISSILE;
    missile.birth_time = 0.0;
    missile.death_time = is_hgv ? 7200.0 : 3600.0;

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

    if (is_hgv) {
        missile.missile_params.cruise_altitude = hgv.cruise_altitude;
        missile.missile_params.glide_ratio = hgv.glide_ratio;
        missile.missile_params.terminal_dive_range = hgv.terminal_dive_range;
        missile.missile_params.pullup_duration = hgv.pullup_duration;
        missile.missile_params.bank_angle_max = hgv.bank_angle_max;
        missile.missile_params.num_skips = hgv.num_skips;
        missile.missile_params.enable_separation = true;
        gen.precomputeHgvTrajectory(missile, false);
    } else {
        gen.precomputeBallisticTrajectory(missile, false);
    }

    if (missile.trajectory_cache.empty()) return {1e12f, 0.0f};

    float max_alt = 0.0f;
    for (const auto& pt : missile.trajectory_cache) {
        if (pt.z > max_alt) max_alt = pt.z;
    }

    const auto& last = missile.trajectory_cache.back();
    float dx = last.x - target_x;
    float dy = last.y - target_y;
    float dist = std::sqrt(dx * dx + dy * dy);
    return {dist, max_alt};
}

// 発射角の最適化（粗→精密の2段階探索）
// target_max_altitude > 0: 着弾閾値内の角度のうち目標高度に最も近い角度を選択
static float optimizeLaunchAngle(
    TargetGenerator& gen,
    float launch_x, float launch_y,
    float target_x, float target_y,
    float boost_duration,
    float initial_mass, float fuel_fraction,
    float specific_impulse, float drag_coefficient,
    float cross_section_area,
    float& out_best_dist,
    float& out_best_alt,
    int& iteration_count,
    float target_max_altitude = 0.0f,
    float distance_threshold = 1e12f,
    bool is_hgv = false,
    const HgvParams& hgv = HgvParams())
{
    // HGVは浅い打上角（0.15-0.55 rad）、弾道は全範囲（0.15-1.40 rad）
    const float ANGLE_MIN = 0.15f;
    const float ANGLE_MAX = is_hgv ? 0.55f : 1.40f;
    const float COARSE_STEP = is_hgv ? 0.02f : 0.03f;
    const float FINE_RANGE = 0.03f;
    const float FINE_STEP = 0.002f;

    float best_angle = is_hgv ? 0.35f : 0.7f;
    float best_dist = 1e12f;
    float best_alt = 0.0f;
    float best_alt_err = 1e12f;

    auto evaluate = [&](float angle) {
        auto res = computeTrajectoryResult(gen, launch_x, launch_y, target_x, target_y,
            angle, boost_duration, initial_mass, fuel_fraction,
            specific_impulse, drag_coefficient, cross_section_area,
            is_hgv, hgv);
        iteration_count++;

        if (target_max_altitude > 0.0f && distance_threshold < 1e11f) {
            if (res.impact_distance <= distance_threshold) {
                float alt_err = std::abs(res.max_altitude - target_max_altitude);
                if (alt_err < best_alt_err || (alt_err == best_alt_err && res.impact_distance < best_dist)) {
                    best_alt_err = alt_err;
                    best_dist = res.impact_distance;
                    best_alt = res.max_altitude;
                    best_angle = angle;
                }
            } else if (best_alt_err > 1e11f && res.impact_distance < best_dist) {
                best_dist = res.impact_distance;
                best_alt = res.max_altitude;
                best_angle = angle;
            }
        } else {
            if (res.impact_distance < best_dist) {
                best_dist = res.impact_distance;
                best_alt = res.max_altitude;
                best_angle = angle;
            }
        }
    };

    // 粗探索
    for (float angle = ANGLE_MIN; angle <= ANGLE_MAX + 0.001f; angle += COARSE_STEP) {
        evaluate(angle);
    }

    // 精密探索: 最良値±FINE_RANGE
    float lo = std::max(ANGLE_MIN, best_angle - FINE_RANGE);
    float hi = std::min(ANGLE_MAX, best_angle + FINE_RANGE);
    for (float angle = lo; angle <= hi + 0.001f; angle += FINE_STEP) {
        evaluate(angle);
    }

    out_best_dist = best_dist;
    out_best_alt = best_alt;
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
    std::cout << "  --radar-min-range <m> Radar minimum range (default: 0)" << std::endl;
    std::cout << "  --radar-max-range <m> Radar max range (0=auto)" << std::endl;
    std::cout << "  --azimuth-coverage <rad> Azimuth coverage width (default: 2π)" << std::endl;
    std::cout << "  --min-elevation <rad>    Minimum elevation angle (default: -0.52)" << std::endl;
    std::cout << "  --max-elevation <rad>    Maximum elevation angle (default: 1.57)" << std::endl;
    std::cout << "  --radar-fov <rad>    [DEPRECATED] Use --azimuth-coverage" << std::endl;
    std::cout << "  --target-max-altitude <m> Target max altitude for auto-adjust (0=auto)" << std::endl;
    std::cout << "  --lock-angle         Lock launch angle (exclude from auto-adjust)" << std::endl;
    std::cout << "  --lock-isp           Lock specific impulse (exclude from auto-adjust)" << std::endl;
    std::cout << "  --lock-fuel          Lock fuel fraction (exclude from auto-adjust)" << std::endl;
    std::cout << "  --num-runs <N>       Number of simulation runs for statistics (default: 1)" << std::endl;
    std::cout << "  --seed <S>           Random seed (0=random, default: 0)" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
}

// ========================================
// trajectory mode: 軌道CSVのみ出力
// ========================================
// キャッシュからdt_output間隔で補間してCSVへ出力するヘルパー
static void outputTrajectoryCSV(
    std::ofstream& traj_file,
    const std::vector<TrajectoryPoint>& cache,
    double dt_output,
    double target_lat, double target_lon,
    int object_id,
    float& max_alt, float& max_speed, int& num_output)
{
    if (cache.empty()) return;
    float flight_duration = static_cast<float>(cache.back().time);
    double last_output_time = -1.0;

    for (double t = cache.front().time; t <= flight_duration + 0.001; t += dt_output) {
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
                  << object_id << ","
                  << std::setprecision(6) << lat << ","
                  << lon << std::endl;

        num_output++;
        last_output_time = tp.time;
    }

    // 最終キャッシュ点（着弾点）が最後の出力時刻と異なる場合、必ず出力する
    // dt_output間隔の補間で着弾点がスキップされるのを防ぐ
    if (cache.back().time > last_output_time + 0.01) {
        const auto& last = cache.back();
        float speed = std::sqrt(last.vx*last.vx + last.vy*last.vy + last.vz*last.vz);
        max_alt = std::max(max_alt, last.z);
        max_speed = std::max(max_speed, speed);
        double lat = 0.0, lon = 0.0;
        metersToLatLon(last.x, last.y, target_lat, target_lon, lat, lon);
        traj_file << std::setprecision(4) << last.time << ","
                  << std::setprecision(2) << last.x << ","
                  << last.y << ","
                  << last.z << ","
                  << last.vx << ","
                  << last.vy << ","
                  << last.vz << ","
                  << std::setprecision(1) << speed << ","
                  << phaseToString(last.phase) << ","
                  << object_id << ","
                  << std::setprecision(6) << lat << ","
                  << lon << std::endl;
        num_output++;
    }
}

static int runTrajectoryMode(
    float launch_x, float launch_y,
    float target_x, float target_y,
    float launch_angle, float boost_duration,
    float initial_mass, float fuel_fraction,
    float specific_impulse, float drag_coefficient,
    float cross_section_area,
    double target_lat, double target_lon,
    double dt_output,
    bool enable_separation = false,
    float warhead_mass_fraction = 0.3f,
    float warhead_cd = 0.15f,
    float booster_cd = 1.5f,
    const std::string& missile_type = "ballistic",
    float cruise_altitude = 40000.0f,
    float glide_ratio = 4.0f,
    float terminal_dive_range = 20000.0f,
    float pullup_duration = 30.0f,
    float bank_angle_max = 0.0f,
    int num_skips = 1)
{
    bool is_hgv = (missile_type == "hgv");

    // TargetParameters設定
    TargetParameters missile;
    missile.motion_model = is_hgv ? MotionModel::HYPERSONIC_GLIDE : MotionModel::BALLISTIC_MISSILE;
    missile.birth_time = 0.0;
    missile.death_time = is_hgv ? 7200.0 : 3600.0;

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
    missile.missile_params.enable_separation = enable_separation;
    missile.missile_params.warhead_mass_fraction = warhead_mass_fraction;
    missile.missile_params.warhead_cd = warhead_cd;
    missile.missile_params.booster_cd = booster_cd;

    if (is_hgv) {
        missile.missile_params.cruise_altitude = cruise_altitude;
        missile.missile_params.glide_ratio = glide_ratio;
        missile.missile_params.terminal_dive_range = terminal_dive_range;
        missile.missile_params.pullup_duration = pullup_duration;
        missile.missile_params.bank_angle_max = bank_angle_max;
        missile.missile_params.num_skips = num_skips;
        missile.missile_params.enable_separation = true;
    }

    // RK4で軌道計算
    TargetGenerator gen(1);
    if (is_hgv) {
        gen.precomputeHgvTrajectory(missile);
    } else {
        gen.precomputeBallisticTrajectory(missile);
    }

    if (missile.trajectory_cache.empty()) {
        std::cerr << "ERROR: Trajectory computation failed" << std::endl;
        return 1;
    }

    // trajectory.csv 出力
    std::ofstream traj_file("trajectory.csv");
    traj_file << "time,x,y,altitude,vx,vy,vz,speed,phase,object_id,lat,lon" << std::endl;
    traj_file << std::fixed;

    float max_alt = 0.0f;
    float max_speed = 0.0f;
    int num_output = 0;

    // 弾頭軌道（object_id=0）
    outputTrajectoryCSV(traj_file, missile.trajectory_cache, dt_output,
                        target_lat, target_lon, 0, max_alt, max_speed, num_output);

    // ブースター軌道（object_id=1）
    if (!missile.booster_trajectory_cache.empty()) {
        float booster_max_alt = 0.0f, booster_max_speed = 0.0f;
        outputTrajectoryCSV(traj_file, missile.booster_trajectory_cache, dt_output,
                            target_lat, target_lon, 1, booster_max_alt, booster_max_speed, num_output);
    }

    traj_file.close();

    float flight_duration = static_cast<float>(missile.trajectory_cache.back().time);
    float range_m = std::sqrt((launch_x - target_x) * (launch_x - target_x) +
                              (launch_y - target_y) * (launch_y - target_y));

    // サマリーをstdoutに出力（FlaskがパースするJSON形式）
    std::cout << std::fixed;
    std::cout << "{" << std::endl;
    std::cout << "  \"range_km\": " << std::setprecision(1) << (range_m / 1000.0f) << "," << std::endl;
    std::cout << "  \"flight_duration\": " << std::setprecision(1) << flight_duration << "," << std::endl;
    std::cout << "  \"max_altitude\": " << std::setprecision(1) << max_alt << "," << std::endl;
    std::cout << "  \"max_speed\": " << std::setprecision(1) << max_speed << "," << std::endl;
    std::cout << "  \"num_steps\": " << num_output;

    if (enable_separation && !missile.booster_trajectory_cache.empty()) {
        float sep_time = static_cast<float>(missile.booster_trajectory_cache.front().time);
        float booster_impact_time = static_cast<float>(missile.booster_trajectory_cache.back().time);
        std::cout << "," << std::endl;
        std::cout << "  \"separation_time\": " << std::setprecision(1) << sep_time << "," << std::endl;
        std::cout << "  \"booster_impact_time\": " << std::setprecision(1) << booster_impact_time;
    }

    std::cout << std::endl;
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
    float distance_threshold_m,
    float target_max_altitude = 0.0f,
    bool lock_angle = false,
    bool lock_isp = false,
    bool lock_fuel = false,
    bool enable_separation = false,
    float warhead_mass_fraction = 0.3f,
    float warhead_cd = 0.15f,
    float booster_cd = 1.5f,
    const std::string& missile_type = "ballistic",
    float cruise_altitude = 40000.0f,
    float glide_ratio = 4.0f,
    float terminal_dive_range = 20000.0f,
    float pullup_duration = 30.0f,
    float bank_angle_max = 0.0f,
    int num_skips = 1)
{
    bool is_hgv = (missile_type == "hgv");
    HgvParams hgv_p{cruise_altitude, glide_ratio, terminal_dive_range, pullup_duration, bank_angle_max, num_skips};

    TargetGenerator gen(1);
    int iterations = 0;

    // 現実的なパラメータ上限
    const float MAX_ISP = is_hgv ? 300.0f : 350.0f;  // HGV: 固体ロケット上限300s, 弾道: 液体350s
    const float MAX_FUEL_FRACTION = 0.85f;  // 構造限界
    const float ISP_STEP = 10.0f;
    const float FF_STEP = 0.02f;

    float best_angle = launch_angle_orig;
    float best_isp = specific_impulse_orig;
    float best_ff = fuel_fraction_orig;
    float best_dist = 1e12f;
    float best_alt = 0.0f;

    // Phase 1: 発射角の最適化
    if (!lock_angle) {
        best_angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
            boost_duration, initial_mass, best_ff, best_isp,
            drag_coefficient, cross_section_area, best_dist, best_alt, iterations,
            target_max_altitude, distance_threshold_m, is_hgv, hgv_p);
    } else {
        auto res = computeTrajectoryResult(gen, launch_x, launch_y, target_x, target_y,
            best_angle, boost_duration, initial_mass, best_ff, best_isp,
            drag_coefficient, cross_section_area, is_hgv, hgv_p);
        best_dist = res.impact_distance;
        best_alt = res.max_altitude;
        iterations++;
    }

    std::cerr << "Phase 1 (angle" << (lock_angle ? " locked" : "") << "): angle=" << best_angle
              << " rad, dist=" << (best_dist / 1000.0f) << " km, alt=" << (best_alt / 1000.0f)
              << " km, iters=" << iterations << std::endl;

    // Phase 2: Isp を段階的に増加
    if (!lock_isp && best_dist > distance_threshold_m && best_isp < MAX_ISP) {
        for (float isp = specific_impulse_orig + ISP_STEP; isp <= MAX_ISP + 0.1f; isp += ISP_STEP) {
            float dist = 0.0f, alt = 0.0f;
            float angle;
            if (!lock_angle) {
                angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
                    boost_duration, initial_mass, best_ff, isp,
                    drag_coefficient, cross_section_area, dist, alt, iterations,
                    target_max_altitude, distance_threshold_m, is_hgv, hgv_p);
            } else {
                angle = best_angle;
                auto res = computeTrajectoryResult(gen, launch_x, launch_y, target_x, target_y,
                    angle, boost_duration, initial_mass, best_ff, isp,
                    drag_coefficient, cross_section_area, is_hgv, hgv_p);
                dist = res.impact_distance;
                alt = res.max_altitude;
                iterations++;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_angle = angle;
                best_isp = isp;
                best_alt = alt;
            }
            if (best_dist <= distance_threshold_m) break;
        }
        std::cerr << "Phase 2 (Isp adj): isp=" << best_isp
                  << ", angle=" << best_angle
                  << ", dist=" << (best_dist / 1000.0f) << " km" << std::endl;
    }

    // Phase 3: 燃料割合を段階的に増加
    if (!lock_fuel && best_dist > distance_threshold_m && best_ff < MAX_FUEL_FRACTION) {
        for (float ff = fuel_fraction_orig + FF_STEP; ff <= MAX_FUEL_FRACTION + 0.001f; ff += FF_STEP) {
            float isp_lo = lock_isp ? specific_impulse_orig : specific_impulse_orig;
            float isp_hi = lock_isp ? specific_impulse_orig : MAX_ISP;
            for (float isp = isp_lo; isp <= isp_hi + 0.1f; isp += ISP_STEP) {
                float dist = 0.0f, alt = 0.0f;
                float angle;
                if (!lock_angle) {
                    angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
                        boost_duration, initial_mass, ff, isp,
                        drag_coefficient, cross_section_area, dist, alt, iterations,
                        target_max_altitude, distance_threshold_m, is_hgv, hgv_p);
                } else {
                    angle = best_angle;
                    auto res = computeTrajectoryResult(gen, launch_x, launch_y, target_x, target_y,
                        angle, boost_duration, initial_mass, ff, isp,
                        drag_coefficient, cross_section_area, is_hgv, hgv_p);
                    dist = res.impact_distance;
                    alt = res.max_altitude;
                    iterations++;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_angle = angle;
                    best_isp = isp;
                    best_ff = ff;
                    best_alt = alt;
                }
                if (lock_isp) break;
            }
            if (best_dist <= distance_threshold_m) break;
        }
        std::cerr << "Phase 3 (fuel adj): ff=" << best_ff
                  << ", isp=" << best_isp
                  << ", angle=" << best_angle
                  << ", dist=" << (best_dist / 1000.0f) << " km" << std::endl;
    }

    // Phase 4 (HGV): 燃料最小化 + Isp現実化
    // HGVは固体ロケットブースターを使用するため、Ispは300s以下が現実的。
    // Phase 2-3で見つけた解をベースに、燃料割合を二分探索で最小化する。
    if (is_hgv && best_dist <= distance_threshold_m) {
        const float HGV_REALISTIC_ISP = 300.0f;

        // 4a: Ispが現実的上限を超えている場合、制限して燃料増で補償
        if (!lock_isp && best_isp > HGV_REALISTIC_ISP) {
            float capped_isp = HGV_REALISTIC_ISP;
            bool capped_ok = false;
            for (float ff = best_ff; ff <= MAX_FUEL_FRACTION + 0.001f; ff += FF_STEP) {
                float dist = 0.0f, alt = 0.0f;
                float angle;
                if (!lock_angle) {
                    angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
                        boost_duration, initial_mass, ff, capped_isp,
                        drag_coefficient, cross_section_area, dist, alt, iterations,
                        target_max_altitude, distance_threshold_m, is_hgv, hgv_p);
                } else {
                    angle = best_angle;
                    auto res = computeTrajectoryResult(gen, launch_x, launch_y, target_x, target_y,
                        angle, boost_duration, initial_mass, ff, capped_isp,
                        drag_coefficient, cross_section_area, is_hgv, hgv_p);
                    dist = res.impact_distance; alt = res.max_altitude;
                    iterations++;
                }
                if (dist <= distance_threshold_m) {
                    best_isp = capped_isp;
                    best_ff = ff;
                    best_angle = angle;
                    best_dist = dist;
                    best_alt = alt;
                    capped_ok = true;
                    break;
                }
            }
            if (!capped_ok) {
                std::cerr << "Phase 4a: Isp=" << HGV_REALISTIC_ISP
                          << "s not viable at max fuel, keeping Isp=" << best_isp << "s" << std::endl;
            }
        }

        // 4b: 燃料割合を二分探索で最小化（確定Ispで到達可能な最小燃料を探す）
        if (!lock_fuel) {
            float ff_lo = 0.20f;
            float ff_hi = best_ff;

            for (int step = 0; step < 15; step++) {
                float ff_mid = (ff_lo + ff_hi) / 2.0f;
                float dist = 0.0f, alt = 0.0f;
                float angle;
                if (!lock_angle) {
                    angle = optimizeLaunchAngle(gen, launch_x, launch_y, target_x, target_y,
                        boost_duration, initial_mass, ff_mid, best_isp,
                        drag_coefficient, cross_section_area, dist, alt, iterations,
                        target_max_altitude, distance_threshold_m, is_hgv, hgv_p);
                } else {
                    angle = best_angle;
                    auto res = computeTrajectoryResult(gen, launch_x, launch_y, target_x, target_y,
                        angle, boost_duration, initial_mass, ff_mid, best_isp,
                        drag_coefficient, cross_section_area, is_hgv, hgv_p);
                    dist = res.impact_distance; alt = res.max_altitude;
                    iterations++;
                }

                if (dist <= distance_threshold_m) {
                    ff_hi = ff_mid;
                    best_ff = ff_mid;
                    best_angle = angle;
                    best_dist = dist;
                    best_alt = alt;
                } else {
                    ff_lo = ff_mid;
                }
            }
        }

        float fuel_mass = initial_mass * best_ff;
        float dry_mass = initial_mass * (1.0f - best_ff);
        std::cerr << "Phase 4 (HGV fuel min): ff=" << best_ff
                  << " (fuel=" << fuel_mass << " kg, dry=" << dry_mass << " kg)"
                  << ", isp=" << best_isp << "s"
                  << ", angle=" << best_angle
                  << ", dist=" << (best_dist / 1000.0f) << " km" << std::endl;
    }

    std::cerr << "Auto-adjust complete: " << iterations << " iterations, "
              << "impact distance=" << (best_dist / 1000.0f) << " km" << std::endl;

    // 最適パラメータで軌道を生成
    TargetParameters missile;
    missile.motion_model = is_hgv ? MotionModel::HYPERSONIC_GLIDE : MotionModel::BALLISTIC_MISSILE;
    missile.birth_time = 0.0;
    missile.death_time = is_hgv ? 7200.0 : 3600.0;

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
    missile.missile_params.enable_separation = enable_separation;
    missile.missile_params.warhead_mass_fraction = warhead_mass_fraction;
    missile.missile_params.warhead_cd = warhead_cd;
    missile.missile_params.booster_cd = booster_cd;

    if (is_hgv) {
        missile.missile_params.cruise_altitude = cruise_altitude;
        missile.missile_params.glide_ratio = glide_ratio;
        missile.missile_params.terminal_dive_range = terminal_dive_range;
        missile.missile_params.pullup_duration = pullup_duration;
        missile.missile_params.bank_angle_max = bank_angle_max;
        missile.missile_params.num_skips = num_skips;
        missile.missile_params.enable_separation = true;
    }

    if (is_hgv) {
        gen.precomputeHgvTrajectory(missile, true);
    } else {
        gen.precomputeBallisticTrajectory(missile, true);
    }

    if (missile.trajectory_cache.empty()) {
        std::cerr << "ERROR: Final trajectory computation failed" << std::endl;
        return 1;
    }

    // trajectory.csv 出力（runTrajectoryModeと同じフォーマット）
    std::ofstream traj_file("trajectory.csv");
    traj_file << "time,x,y,altitude,vx,vy,vz,speed,phase,object_id,lat,lon" << std::endl;
    traj_file << std::fixed;

    float max_alt = 0.0f;
    float max_speed = 0.0f;
    int num_output = 0;

    // 弾頭軌道（object_id=0）
    outputTrajectoryCSV(traj_file, missile.trajectory_cache, dt_output,
                        target_lat, target_lon, 0, max_alt, max_speed, num_output);

    // ブースター軌道（object_id=1）
    if (!missile.booster_trajectory_cache.empty()) {
        float booster_max_alt = 0.0f, booster_max_speed = 0.0f;
        outputTrajectoryCSV(traj_file, missile.booster_trajectory_cache, dt_output,
                            target_lat, target_lon, 1, booster_max_alt, booster_max_speed, num_output);
    }

    traj_file.close();

    float flight_duration = static_cast<float>(missile.trajectory_cache.back().time);
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
    std::cout << "  \"adj_fuel_mass\": " << std::setprecision(1) << (initial_mass * best_ff) << "," << std::endl;
    std::cout << "  \"impact_distance_km\": " << std::setprecision(1) << (best_dist / 1000.0f) << "," << std::endl;
    std::cout << "  \"search_iterations\": " << iterations;

    if (enable_separation && !missile.booster_trajectory_cache.empty()) {
        float sep_time = static_cast<float>(missile.booster_trajectory_cache.front().time);
        float booster_impact_time = static_cast<float>(missile.booster_trajectory_cache.back().time);
        std::cout << "," << std::endl;
        std::cout << "  \"separation_time\": " << std::setprecision(1) << sep_time << "," << std::endl;
        std::cout << "  \"booster_impact_time\": " << std::setprecision(1) << booster_impact_time;
    }

    std::cout << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}

// ========================================
// CLI上書き用グローバル変数（-1=シナリオデフォルト使用）
// ========================================
static float cli_gate_threshold = -1.0f;
static int   cli_confirm_hits = -1;
static int   cli_confirm_window = -1;
static int   cli_delete_misses = -1;
static float cli_min_snr = -1001.0f;  // sentinel: <= -1000 means "not set"
static float cli_process_pos = -1.0f;
static float cli_process_vel = -1.0f;
static float cli_process_acc = -1.0f;
static float cli_detect_prob = -1.0f;
// JPDA パラメータ
static std::string cli_association_method = "gnn";
static float cli_jpda_pd = -1.0f;
static float cli_jpda_clutter_density = -1.0f;
static float cli_jpda_gate = -1.0f;
// PMBM パラメータ
static float cli_pmbm_pd = -1.0f;
static float cli_pmbm_gate = -1.0f;
static float cli_pmbm_clutter_density = -1.0f;
static int   cli_pmbm_k_best = -1;
static float cli_pmbm_survival = -1.0f;
static float cli_pmbm_init_existence = -1.0f;
// MHT parameters
static float cli_mht_pd = -1.0f;
static float cli_mht_gate = -1.0f;
static float cli_mht_clutter_density = -1.0f;
static int   cli_mht_k_best = -1;
static int   cli_mht_max_hypotheses = -1;
static float cli_mht_score_decay = -1.0f;
static float cli_mht_prune_ratio = -1.0f;
static float cli_mht_switch_cost = -1.0f;
// GLMB parameters
static float cli_glmb_pd = -1.0f;
static float cli_glmb_gate = -1.0f;
static float cli_glmb_clutter_density = -1.0f;
static int   cli_glmb_k_best = -1;
static int   cli_glmb_max_hypotheses = -1;
static float cli_glmb_survival = -1.0f;
static float cli_glmb_birth_weight = -1.0f;
static float cli_glmb_score_decay = -1.0f;
static float cli_glmb_init_existence = -1.0f;
static std::string cli_glmb_sampler = "";
static int   cli_glmb_gibbs_sweeps = -1;
static int   cli_glmb_gibbs_burnin = -1;
// UKF パラメータ
static float cli_ukf_alpha = -1.0f;
static float cli_ukf_beta = -1.0f;
static float cli_ukf_kappa = -999.0f;  // kappaは負値も有効なので-999を無効値に
static float cli_max_distance = -1.0f;
static float cli_max_jump_velocity = -1.0f;
static float cli_min_init_distance = -1.0f;
// IMM 遷移確率行列 (4×4: CA, Ballistic, CT, SkipGlide)
static float cli_imm_ca_ca = -1.0f;
static float cli_imm_ca_bal = -1.0f;
static float cli_imm_ca_ct = -1.0f;
static float cli_imm_ca_sg = -1.0f;
static float cli_imm_bal_ca = -1.0f;
static float cli_imm_bal_bal = -1.0f;
static float cli_imm_bal_ct = -1.0f;
static float cli_imm_bal_sg = -1.0f;
static float cli_imm_ct_ca = -1.0f;
static float cli_imm_ct_bal = -1.0f;
static float cli_imm_ct_ct = -1.0f;
static float cli_imm_ct_sg = -1.0f;
static float cli_imm_sg_ca = -1.0f;
static float cli_imm_sg_bal = -1.0f;
static float cli_imm_sg_ct = -1.0f;
static float cli_imm_sg_sg = -1.0f;
// IMM モデルノイズ倍率
static float cli_imm_ca_noise = -1.0f;
static float cli_imm_bal_noise = -1.0f;
static float cli_imm_ct_noise = -1.0f;
static float cli_imm_sg_noise = -1.0f;
// センサーパラメータ
static float cli_range_noise = -1.0f;
static float cli_azimuth_noise = -1.0f;
static float cli_elevation_noise = -1.0f;
static float cli_doppler_noise = -1.0f;
static float cli_pfa = -1.0f;
static float cli_snr_ref = -1.0f;
static float cli_pd_ref = -1.0f;        // 基準距離における検出確率（-1=自動）
static float cli_pd_ref_range = -1.0f;  // 検出性能基準距離 [m]（-1=デフォルト10km）

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
    float radar_min_range, float radar_max_range, float radar_fov,
    float azimuth_coverage = 2.0f * M_PI,
    float min_elevation = -0.5236f,
    float max_elevation = 1.5708f,
    bool enable_separation = false,
    float warhead_mass_fraction = 0.3f,
    float warhead_cd = 0.15f,
    float booster_cd = 1.5f,
    int num_runs = 1,
    uint32_t seed = 0,
    const std::string& missile_type = "ballistic",
    float cruise_altitude = 40000.0f,
    float glide_ratio = 4.0f,
    float terminal_dive_range = 20000.0f,
    float pullup_duration = 30.0f,
    float bank_angle_max = 0.0f,
    int num_skips = 1,
    int cluster_count = 0,
    float cluster_spread = 5000.0f,
    float launch_time_spread = 5.0f,
    float beam_width = 0.052f,
    int num_beams = 10,
    int min_search_beams = 1,
    bool track_confirmed_only = false,
    float search_sector = -1.0f,
    float search_center = 0.0f,
    float antenna_boresight = 0.0f,
    float search_min_range = 0.0f,
    float search_max_range = 0.0f,
    float track_range_width = 0.0f,
    float range_resolution = 150.0f,
    float elev_scan_min_arg = 0.0f,
    float elev_scan_max_arg = 0.35f,
    int elev_bars_per_frame_arg = 3,
    int elev_cycle_steps_arg = 9)
{
    // CUDA デバイス情報（GPU利用可能性チェック）
    bool gpu_available = false;
    try {
        auto device_info = cuda::DeviceInfo::getCurrent();
        device_info.print();
        gpu_available = true;
        std::cout << "\n✓ GPU Available - Acceleration enabled\n" << std::endl;
    } catch (const std::exception& e) {
        // GPU利用不可の場合、エラーをログに出力してCPU専用モードで続行
        std::ofstream error_log("/tmp/fasttracker_gpu_error.log");
        error_log << "========================================\n";
        error_log << "FastTracker GPU Error\n";
        error_log << "========================================\n";
        error_log << "Error: " << e.what() << "\n";
        error_log << "\nThis error typically means:\n";
        error_log << "- CUDA driver version is older than runtime (12.6.0)\n";
        error_log << "- GPU is not available in this environment\n";
        error_log << "- Running in WSL2 without CUDA support\n";
        error_log << "\nSolution:\n";
        error_log << "The tracker will automatically run in CPU-only mode.\n";
        error_log << "GPU acceleration will be disabled (tracking still works).\n";
        error_log << "========================================\n";
        error_log.close();

        std::cerr << "\n";
        std::cerr << "========================================\n";
        std::cerr << "⚠️  GPU UNAVAILABLE - Running in CPU-only mode\n";
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Details saved to: /tmp/fasttracker_gpu_error.log\n";
        std::cerr << "========================================\n";
        std::cerr << "\n";
        gpu_available = false;
        // CPU専用モードで続行（return 1を削除）
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
        bool is_hgv = (missile_type == "hgv");
        float range_m = std::sqrt((launch_x - target_x) * (launch_x - target_x) +
                                  (launch_y - target_y) * (launch_y - target_y));
        std::cout << (is_hgv ? "Single HGV" : "Single Ballistic Missile")
                  << " Scenario (" << (range_m / 1000.0f) << "km range)" << std::endl;

        TargetParameters missile;
        missile.motion_model = is_hgv ? MotionModel::HYPERSONIC_GLIDE : MotionModel::BALLISTIC_MISSILE;
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
        missile.missile_params.enable_separation = enable_separation;
        missile.missile_params.warhead_mass_fraction = warhead_mass_fraction;
        missile.missile_params.warhead_cd = warhead_cd;
        missile.missile_params.booster_cd = booster_cd;

        if (is_hgv) {
            missile.missile_params.cruise_altitude = cruise_altitude;
            missile.missile_params.glide_ratio = glide_ratio;
            missile.missile_params.terminal_dive_range = terminal_dive_range;
            missile.missile_params.pullup_duration = pullup_duration;
            missile.missile_params.bank_angle_max = bank_angle_max;
            missile.missile_params.num_skips = num_skips;
            missile.missile_params.enable_separation = true;
        }

        // 軌道を事前計算
        if (is_hgv) {
            target_gen.precomputeHgvTrajectory(missile);
        } else {
            target_gen.precomputeBallisticTrajectory(missile);
        }

        std::vector<TargetParameters> params_vec;
        params_vec.push_back(missile);

        // クラスタ目標生成（メイン弾頭の周辺に追加目標を配置）
        if (cluster_count > 0) {
            std::mt19937 cluster_rng(seed > 0 ? (seed + 42) : std::random_device{}());
            std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
            std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
            std::normal_distribution<float> normal(0.0f, 1.0f);

            std::cerr << "  Generating " << cluster_count << " cluster targets (spread: "
                      << (cluster_spread / 1000.0f) << " km, time spread: "
                      << launch_time_spread << " s)" << std::endl;

            for (int ci = 0; ci < cluster_count; ci++) {
                TargetParameters cm = missile;  // メインのパラメータをコピー

                // 発射位置のみ摂動（他パラメータは基準軌道と同一）
                float a = uniform(cluster_rng) * static_cast<float>(M_PI);
                float r = std::abs(normal(cluster_rng)) * cluster_spread * 0.5f;
                cm.initial_state(0) = launch_x + r * std::cos(a);
                cm.initial_state(1) = launch_y + r * std::sin(a);

                // 軌道キャッシュクリア→再計算
                cm.trajectory_cache.clear();
                cm.booster_trajectory_cache.clear();
                cm.trajectory_computed = false;

                if (is_hgv) {
                    target_gen.precomputeHgvTrajectory(cm, false);
                } else {
                    target_gen.precomputeBallisticTrajectory(cm, false);
                }

                if (!cm.trajectory_cache.empty()) {
                    params_vec.push_back(cm);
                } else {
                    std::cerr << "  Warning: cluster target " << ci << " trajectory failed, skipping" << std::endl;
                }
            }
            std::cerr << "  Cluster targets generated: " << (params_vec.size() - 1) << std::endl;
        }

        // 分離時: 全弾頭のブースターを追加
        if (enable_separation) {
            int n_warheads = static_cast<int>(params_vec.size());
            for (int wi = 0; wi < n_warheads; wi++) {
                auto& warhead = params_vec[wi];
                if (warhead.booster_trajectory_cache.empty()) continue;

                TargetParameters booster;
                booster.motion_model = MotionModel::BALLISTIC_MISSILE;
                booster.birth_time = warhead.booster_trajectory_cache.front().time + warhead.birth_time;
                booster.death_time = warhead.booster_trajectory_cache.back().time + warhead.birth_time;
                booster.initial_state.setZero();
                booster.initial_state(0) = warhead.booster_trajectory_cache.front().x;
                booster.initial_state(1) = warhead.booster_trajectory_cache.front().y;
                booster.initial_state(2) = warhead.booster_trajectory_cache.front().z;
                booster.initial_state(3) = warhead.booster_trajectory_cache.front().vx;
                booster.initial_state(4) = warhead.booster_trajectory_cache.front().vy;
                booster.initial_state(5) = warhead.booster_trajectory_cache.front().vz;
                booster.trajectory_cache = warhead.booster_trajectory_cache;
                double booster_t0 = booster.birth_time;
                for (auto& tp : booster.trajectory_cache) {
                    tp.time -= booster_t0;
                }
                booster.trajectory_computed = true;
                booster.missile_params = warhead.missile_params;
                params_vec.push_back(booster);
            }
            std::cout << "  Separation enabled: " << n_warheads << " warhead(s) + "
                      << (params_vec.size() - n_warheads) << " booster(s)" << std::endl;
        }

        num_targets = static_cast<int>(params_vec.size());
        target_gen.initializeCustomScenario(params_vec);

        // durationを実際の飛翔時間に更新（全目標の最大着弾時刻）
        auto& target_params = const_cast<std::vector<TargetParameters>&>(target_gen.getTargetParams());
        duration = 0.0;
        for (size_t i = 0; i < target_params.size(); i++) {
            if (target_params[i].death_time > duration) {
                duration = target_params[i].death_time;
            }
        }
        if (duration > 0.0) {
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

        radar_params.min_range = radar_min_range;

        if (radar_max_range > 0.0f) {
            radar_params.max_range = radar_max_range;
        } else {
            radar_params.max_range = max_sensor_dist * 1.3f;
        }

        radar_params.azimuth_coverage = azimuth_coverage;
        radar_params.min_elevation = min_elevation;
        radar_params.max_elevation = max_elevation;
        radar_params.field_of_view = azimuth_coverage;  // Backward compatibility

        radar_params.false_alarm_rate = 0.1f / (M_PI * radar_params.max_range * radar_params.max_range);

        std::cout << "  Sensor position: (" << (sensor_x / 1000.0f) << "km, " << (sensor_y / 1000.0f) << "km)" << std::endl;
        std::cout << "  Radar range: " << (radar_params.min_range / 1000.0f) << " - " << (radar_params.max_range / 1000.0f) << " km" << std::endl;
    } else if (scenario == "ballistic" || scenario == "hypersonic" || scenario == "mixed-threat") {
        radar_params.max_range = 150000.0f;
        radar_params.false_alarm_rate = 1e-10f;
    }

    // CLI上書き: センサーパラメータ
    if (cli_detect_prob >= 0) radar_params.detection_probability = cli_detect_prob;
    if (cli_range_noise >= 0) radar_params.meas_noise.range_noise = cli_range_noise;
    if (cli_azimuth_noise >= 0) radar_params.meas_noise.azimuth_noise = cli_azimuth_noise;
    if (cli_elevation_noise >= 0) radar_params.meas_noise.elevation_noise = cli_elevation_noise;
    if (cli_doppler_noise >= 0) radar_params.meas_noise.doppler_noise = cli_doppler_noise;
    if (cli_pfa >= 0) {
        // Pfa → クラッタ密度変換
        float dr = radar_params.meas_noise.range_noise;
        float dtheta = radar_params.meas_noise.azimuth_noise;
        float cell_area = dr * dtheta * radar_params.max_range * 0.5f;
        if (cell_area > 0.0f) {
            radar_params.false_alarm_rate = cli_pfa / cell_area;
        }
        radar_params.pfa_per_pulse = cli_pfa;
    }
    if (cli_pd_ref >= 0)       radar_params.pd_at_ref_range = cli_pd_ref;
    if (cli_pd_ref_range >= 0) radar_params.det_ref_range_m = cli_pd_ref_range;

    // SNR Ref: --snr-ref が明示指定された場合のみ直接上書き、それ以外は P_FA / R_max / P(D) から自動計算
    if (cli_snr_ref >= 0) {
        radar_params.snr_ref = cli_snr_ref;
    } else {
        radar_params.computeSnrRef();
    }

    // ビームステアリング
    radar_params.beam_width = beam_width;
    radar_params.num_beams = num_beams;
    radar_params.antenna_boresight = antenna_boresight;

    // 方位覆域範囲 (boresight中心)
    float half_azimuth_cov = radar_params.azimuth_coverage / 2.0f;
    float cov_min_az = antenna_boresight - half_azimuth_cov;
    float cov_max_az = antenna_boresight + half_azimuth_cov;

    // サーチセクタの実効範囲を計算
    float search_half = (search_sector > 0.0f)
        ? search_sector / 2.0f
        : half_azimuth_cov;
    float search_min_az = search_center - search_half;
    float search_max_az = search_center + search_half;

    // サーチ領域の距離範囲を設定
    if (search_max_range <= 0.0f) {
        search_max_range = radar_params.max_range;  // デフォルトはレーダー最大距離
    }
    if (search_min_range < 0.0f) {
        search_min_range = 0.0f;
    }
    if (search_min_range >= search_max_range) {
        std::cerr << "Warning: search_min_range (" << search_min_range
                  << ") >= search_max_range (" << search_max_range
                  << "). Setting search_min_range to 0." << std::endl;
        search_min_range = 0.0f;
    }

    std::cout << "  Antenna boresight: " << (antenna_boresight * 180.0 / M_PI) << " deg"
              << ", Azimuth Coverage=[" << (cov_min_az * 180.0 / M_PI)
              << "," << (cov_max_az * 180.0 / M_PI) << "] deg"
              << ", Elevation Range=[" << (radar_params.min_elevation * 180.0 / M_PI)
              << "," << (radar_params.max_elevation * 180.0 / M_PI) << "] deg" << std::endl;
    std::cout << "  Beam steering: " << num_beams << " beams, width="
              << (beam_width * 180.0 / M_PI) << " deg, min_search="
              << min_search_beams << ", priority="
              << (track_confirmed_only ? "confirmed" : "all")
              << ", search=[" << (search_min_az * 180.0 / M_PI)
              << "," << (search_max_az * 180.0 / M_PI) << "] deg"
              << std::endl;
    std::cout << "  Search range: " << (search_min_range / 1000.0f) << " - "
              << (search_max_range / 1000.0f) << " km" << std::endl;

    // ビームリソース警告
    if (num_beams <= min_search_beams) {
        std::cerr << "\n";
        std::cerr << "========================================" << std::endl;
        std::cerr << "WARNING: Insufficient beams for tracking!" << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "  Num Beams: " << num_beams << std::endl;
        std::cerr << "  Min Search Beams: " << min_search_beams << std::endl;
        std::cerr << "  Available for tracking: " << std::max(0, num_beams - min_search_beams) << std::endl;
        std::cerr << "\n";
        std::cerr << "  ** NO TRACK BEAMS WILL BE ALLOCATED **" << std::endl;
        std::cerr << "\n";
        std::cerr << "  To fix: Increase num_beams OR decrease min_search_beams" << std::endl;
        std::cerr << "  Recommended: num_beams >= 30, min_search_beams <= 5" << std::endl;
        std::cerr << "========================================\n" << std::endl;
    }

    // レーダーパラメータにサーチ範囲を設定
    radar_params.search_min_range = search_min_range;
    radar_params.search_max_range = search_max_range;
    radar_params.track_range_width = track_range_width;
    radar_params.range_resolution = range_resolution;

    // トラッカーパラメータ
    ProcessNoise process_noise;
    AssociationParams assoc_params;

    if (scenario == "single-ballistic") {
        process_noise.position_noise = 300.0f;
        process_noise.velocity_noise = 150.0f;
        process_noise.accel_noise = 300.0f;

        assoc_params.gate_threshold = 10.0f;
        assoc_params.max_distance = 100000.0f;
        assoc_params.confirm_hits = 4;              // 確認ヒット数を増加（3→4）: FA起源の誤確定を抑制
        assoc_params.confirm_window = 12;            // 確認窓も拡大（10→12）: ヒット収集猶予を確保
        assoc_params.delete_misses = 60;             // 消失判定を緩和（15→60=2秒@30Hz）: トラック生存延長
        assoc_params.min_snr_for_init = 12.0f;
        assoc_params.min_init_distance = 20000.0f;  // 20km以内の既存航跡があれば新規生成しない
    } else if (scenario == "hypersonic") {
        // HGV専用: 航跡分裂完全防止のため、極めて緩いパラメータ
        // 反復改善2回目: gate=50, confirm=2, delete=40でも分裂 → さらに緩和
        process_noise.position_noise = 20.0f;  // 極端な予測誤差を許容
        process_noise.velocity_noise = 30.0f;  // 極端な速度変化を許容
        process_noise.accel_noise = 25.0f;    // 極端な加速度変化を許容

        assoc_params.gate_threshold = 80.0f;   // 予測誤差許容を最大限拡大（50→80）
        assoc_params.max_distance = 5.0f;
        assoc_params.confirm_hits = 1;         // 即時確定（2→1）
        assoc_params.confirm_window = 5;
        assoc_params.delete_misses = 60;       // 追跡ロス耐性を極限強化（40→60）
        assoc_params.min_snr_for_init = 40.0f;
    } else if (scenario == "ballistic" || scenario == "mixed-threat") {
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

    // CLI上書き（指定された場合のみシナリオデフォルトを上書き）
    if (cli_gate_threshold >= 0) assoc_params.gate_threshold = cli_gate_threshold;
    if (cli_confirm_hits >= 0) assoc_params.confirm_hits = cli_confirm_hits;
    if (cli_confirm_window >= 0) assoc_params.confirm_window = cli_confirm_window;
    if (cli_delete_misses >= 0) assoc_params.delete_misses = cli_delete_misses;
    if (cli_min_snr > -1000.0f) assoc_params.min_snr_for_init = cli_min_snr;  // Allow negative SNR thresholds
    if (cli_max_distance >= 0) assoc_params.max_distance = cli_max_distance;
    if (cli_max_jump_velocity >= 0) assoc_params.max_jump_velocity = cli_max_jump_velocity;
    if (cli_min_init_distance >= 0) assoc_params.min_init_distance = cli_min_init_distance;
    // JPDA パラメータ上書き
    if (cli_association_method == "jpda") {
        assoc_params.association_method = AssociationMethod::JPDA;
    } else if (cli_association_method == "pmbm") {
        assoc_params.association_method = AssociationMethod::PMBM;
    } else if (cli_association_method == "mht") {
        assoc_params.association_method = AssociationMethod::MHT;
    } else if (cli_association_method == "glmb") {
        assoc_params.association_method = AssociationMethod::GLMB;
    }
    if (cli_jpda_pd >= 0) assoc_params.jpda_pd = cli_jpda_pd;
    if (cli_jpda_clutter_density >= 0) assoc_params.jpda_clutter_density = cli_jpda_clutter_density;
    if (cli_jpda_gate >= 0) assoc_params.jpda_gate = cli_jpda_gate;
    // PMBM パラメータ上書き
    if (cli_pmbm_pd >= 0) assoc_params.pmbm_pd = cli_pmbm_pd;
    if (cli_pmbm_gate >= 0) assoc_params.pmbm_gate = cli_pmbm_gate;
    if (cli_pmbm_clutter_density >= 0) assoc_params.pmbm_clutter_density = cli_pmbm_clutter_density;
    if (cli_pmbm_k_best >= 0) assoc_params.pmbm_k_best = cli_pmbm_k_best;
    if (cli_pmbm_survival >= 0) assoc_params.pmbm_survival_prob = cli_pmbm_survival;
    if (cli_pmbm_init_existence >= 0) assoc_params.pmbm_initial_existence = cli_pmbm_init_existence;
    // MHT parameters
    if (cli_mht_pd >= 0) assoc_params.mht_pd = cli_mht_pd;
    if (cli_mht_gate >= 0) assoc_params.mht_gate = cli_mht_gate;
    if (cli_mht_clutter_density >= 0) assoc_params.mht_clutter_density = cli_mht_clutter_density;
    if (cli_mht_k_best >= 0) assoc_params.mht_k_best = cli_mht_k_best;
    if (cli_mht_max_hypotheses >= 0) assoc_params.mht_max_hypotheses = cli_mht_max_hypotheses;
    if (cli_mht_score_decay >= 0) assoc_params.mht_score_decay = cli_mht_score_decay;
    if (cli_mht_prune_ratio >= 0) assoc_params.mht_prune_ratio = cli_mht_prune_ratio;
    if (cli_mht_switch_cost >= 0) assoc_params.mht_switch_cost = cli_mht_switch_cost;
    // GLMB parameters
    if (cli_glmb_pd >= 0) assoc_params.glmb_pd = cli_glmb_pd;
    if (cli_glmb_gate >= 0) assoc_params.glmb_gate = cli_glmb_gate;
    if (cli_glmb_clutter_density >= 0) assoc_params.glmb_clutter_density = cli_glmb_clutter_density;
    if (cli_glmb_k_best >= 0) assoc_params.glmb_k_best = cli_glmb_k_best;
    if (cli_glmb_max_hypotheses >= 0) assoc_params.glmb_max_hypotheses = cli_glmb_max_hypotheses;
    if (cli_glmb_survival >= 0) assoc_params.glmb_survival_prob = cli_glmb_survival;
    if (cli_glmb_birth_weight >= 0) assoc_params.glmb_birth_weight = cli_glmb_birth_weight;
    if (cli_glmb_score_decay >= 0) assoc_params.glmb_score_decay = cli_glmb_score_decay;
    if (cli_glmb_init_existence >= 0) assoc_params.glmb_initial_existence = cli_glmb_init_existence;
    if (cli_glmb_sampler == "gibbs") assoc_params.glmb_sampler = GLMBSampler::GIBBS;
    else if (cli_glmb_sampler == "murty") assoc_params.glmb_sampler = GLMBSampler::MURTY;
    if (cli_glmb_gibbs_sweeps >= 0) assoc_params.glmb_gibbs_sweeps = cli_glmb_gibbs_sweeps;
    if (cli_glmb_gibbs_burnin >= 0) assoc_params.glmb_gibbs_burnin = cli_glmb_gibbs_burnin;
    if (cli_process_pos >= 0) process_noise.position_noise = cli_process_pos;
    if (cli_process_vel >= 0) process_noise.velocity_noise = cli_process_vel;
    if (cli_process_acc >= 0) process_noise.accel_noise = cli_process_acc;

    // UKF パラメータ上書き
    UKFParams ukf_params;
    if (cli_ukf_alpha >= 0) ukf_params.alpha = cli_ukf_alpha;
    if (cli_ukf_beta >= 0) ukf_params.beta = cli_ukf_beta;
    if (cli_ukf_kappa > -999.0f) ukf_params.kappa = cli_ukf_kappa;  // kappa can be negative
    // Recompute lambda after parameter changes
    ukf_params.lambda = ukf_params.alpha * ukf_params.alpha * (STATE_DIM + ukf_params.kappa) - STATE_DIM;

    // max_tracks must accommodate: actual targets, clutter-generated tentative tracks,
    // and multiple detections per target from beam steering (each beam can detect the target)
    // Estimate expected clutter per frame based on Pfa and scan area
    // FOV sector area = 0.5 * FOV_angle * max_range^2
    // Note: false_alarm_rate is already per unit area (Pfa / cell_area), so we multiply by total area
    float fov_area = 0.5f * radar_params.field_of_view * radar_params.max_range * radar_params.max_range;
    int expected_clutter_per_frame = static_cast<int>(radar_params.false_alarm_rate * fov_area * num_beams) + 10;

    int max_tracks = std::max({
        num_targets * 2,
        10,
        num_beams * 3,
        (1 + cluster_count) * num_beams,
        expected_clutter_per_frame + num_targets + 10  // clutter + targets + margin
    });

    // Cap max_tracks to prevent GPU memory exhaustion
    // Typical GPU (8GB) can handle ~2000-5000 tracks depending on state dimension
    constexpr int MAX_TRACKS_LIMIT = 2000;
    if (max_tracks > MAX_TRACKS_LIMIT) {
        std::cerr << "WARNING: Calculated max_tracks (" << max_tracks
                  << ") exceeds GPU memory limit. Capping at " << MAX_TRACKS_LIMIT << std::endl;
        std::cerr << "  This may cause track drops when clutter density is high." << std::endl;
        std::cerr << "  Recommendation: Reduce Pfa to <= 1e-6 for this scenario." << std::endl;
        max_tracks = MAX_TRACKS_LIMIT;
    }

    std::cerr << "[DEBUG] Max tracks calculation:" << std::endl;
    std::cerr << "  Expected clutter per frame: " << expected_clutter_per_frame << std::endl;
    std::cerr << "  FOV area: " << fov_area << " m²" << std::endl;
    std::cerr << "  False alarm rate: " << radar_params.false_alarm_rate << std::endl;
    std::cerr << "  Num beams: " << num_beams << std::endl;
    std::cerr << "  Computed max_tracks: " << max_tracks << std::endl;
    std::cerr << std::flush;
    float ospa_cutoff = (scenario == "single-ballistic") ? 10000.0f : 100.0f;

    double dt = 1.0 / framerate;
    int num_frames = static_cast<int>(duration / dt);

    // Output resolved parameters (for GUI display of auto values)
    // Reverse-compute Pfa from false_alarm_rate
    float resolved_pfa = 0.0f;
    {
        float dr = radar_params.meas_noise.range_noise;
        float dtheta = radar_params.meas_noise.azimuth_noise;
        float cell_area = dr * dtheta * radar_params.max_range * 0.5f;
        if (cell_area > 0.0f) resolved_pfa = radar_params.false_alarm_rate * cell_area;
    }
    // Calculate SNR at reference distance for GUI display
    float gamma_T = -std::log(std::max(radar_params.pfa_per_pulse, 1e-30f));
    float pd_clamped = std::max(std::min(radar_params.pd_at_ref_range, 0.9999f), 1e-4f);
    float snr_avg_lin = gamma_T / (-std::log(pd_clamped));
    float snr_at_ref_range = 10.0f * std::log10(std::max(snr_avg_lin, 1e-10f));

    std::cout << "\n[Resolved Parameters]" << std::endl;
    std::cout << "  radar_max_range: " << radar_params.max_range << std::endl;
    std::cout << "  snr_ref: " << radar_params.snr_ref << std::endl;
    std::cout << "  snr_at_ref_range: " << snr_at_ref_range << std::endl;
    std::cout << "  pd_at_ref_range: " << radar_params.pd_at_ref_range << std::endl;
    std::cout << "  det_ref_range_m: " << radar_params.det_ref_range_m << std::endl;
    std::cout << "  pfa: " << resolved_pfa << std::endl;
    std::cout << "  detect_prob: " << radar_params.detection_probability << std::endl;
    std::cout << "  range_noise: " << radar_params.meas_noise.range_noise << std::endl;
    std::cout << "  azimuth_noise: " << radar_params.meas_noise.azimuth_noise << std::endl;
    std::cout << "  elevation_noise: " << radar_params.meas_noise.elevation_noise << std::endl;
    std::cout << "  doppler_noise: " << radar_params.meas_noise.doppler_noise << std::endl;
    std::cout << "  gate_threshold: " << assoc_params.gate_threshold << std::endl;
    std::cout << "  confirm_hits: " << assoc_params.confirm_hits << std::endl;
    std::cout << "  delete_misses: " << assoc_params.delete_misses << std::endl;
    std::cout << "  min_snr: " << assoc_params.min_snr_for_init << std::endl;
    std::cout << "  max_jump_velocity: " << assoc_params.max_jump_velocity << " m/s" << std::endl;
    std::cout << "  min_init_distance: " << assoc_params.min_init_distance << " m" << std::endl;
    std::cout << "  association: " << (assoc_params.association_method == AssociationMethod::GLMB ? "GLMB" :
                                        assoc_params.association_method == AssociationMethod::MHT ? "MHT" :
                                        assoc_params.association_method == AssociationMethod::PMBM ? "PMBM" :
                                        assoc_params.association_method == AssociationMethod::JPDA ? "JPDA" : "GNN") << std::endl;
    if (assoc_params.association_method == AssociationMethod::JPDA) {
        std::cout << "  jpda_pd: " << assoc_params.jpda_pd << std::endl;
        std::cout << "  jpda_clutter_density: " << assoc_params.jpda_clutter_density << std::endl;
        std::cout << "  jpda_gate: " << assoc_params.jpda_gate << std::endl;
    }
    if (assoc_params.association_method == AssociationMethod::MHT) {
        std::cout << "  mht_pd: " << assoc_params.mht_pd << std::endl;
        std::cout << "  mht_gate: " << assoc_params.mht_gate << std::endl;
        std::cout << "  mht_clutter_density: " << assoc_params.mht_clutter_density << std::endl;
        std::cout << "  mht_k_best: " << assoc_params.mht_k_best << std::endl;
        std::cout << "  mht_max_hypotheses: " << assoc_params.mht_max_hypotheses << std::endl;
        std::cout << "  mht_score_decay: " << assoc_params.mht_score_decay << std::endl;
        std::cout << "  mht_prune_ratio: " << assoc_params.mht_prune_ratio << std::endl;
        std::cout << "  mht_switch_cost: " << assoc_params.mht_switch_cost << std::endl;
    }
    if (assoc_params.association_method == AssociationMethod::GLMB) {
        std::cout << "  glmb_pd: " << assoc_params.glmb_pd << std::endl;
        std::cout << "  glmb_gate: " << assoc_params.glmb_gate << std::endl;
        std::cout << "  glmb_clutter_density: " << assoc_params.glmb_clutter_density << std::endl;
        std::cout << "  glmb_k_best: " << assoc_params.glmb_k_best << std::endl;
        std::cout << "  glmb_max_hypotheses: " << assoc_params.glmb_max_hypotheses << std::endl;
        std::cout << "  glmb_survival_prob: " << assoc_params.glmb_survival_prob << std::endl;
        std::cout << "  glmb_birth_weight: " << assoc_params.glmb_birth_weight << std::endl;
        std::cout << "  glmb_score_decay: " << assoc_params.glmb_score_decay << std::endl;
        std::cout << "  glmb_initial_existence: " << assoc_params.glmb_initial_existence << std::endl;
        std::cout << "  glmb_sampler: " << (assoc_params.glmb_sampler == GLMBSampler::GIBBS ? "gibbs" : "murty") << std::endl;
        if (assoc_params.glmb_sampler == GLMBSampler::GIBBS) {
            std::cout << "  glmb_gibbs_sweeps: " << assoc_params.glmb_gibbs_sweeps << std::endl;
            std::cout << "  glmb_gibbs_burnin: " << assoc_params.glmb_gibbs_burnin << std::endl;
        }
    }
    std::cout << "  process_pos_noise: " << process_noise.position_noise << std::endl;
    std::cout << "  process_vel_noise: " << process_noise.velocity_noise << std::endl;
    std::cout << "  process_acc_noise: " << process_noise.accel_noise << std::endl;

    if (num_runs > 1) {
        std::cout << "\nMulti-run mode: " << num_runs << " runs"
                  << " (seed=" << seed << ")" << std::endl;
    }

    // メトリクス蓄積用
    std::vector<AccuracyMetrics> all_accuracy(num_runs);
    std::vector<DetectionMetrics> all_detection(num_runs);
    std::vector<double> all_wall_time_ms(num_runs);
    std::vector<double> all_avg_frame_ms(num_runs);

    for (int run = 0; run < num_runs; run++) {
        // シード設定
        uint32_t run_seed = (seed > 0) ? (seed + run) : 0;
        if (run_seed > 0) {
            target_gen.setSeed(run_seed);
        }

        // RadarSimulator（毎run新規構築）
        RadarSimulator radar_sim(target_gen, radar_params);
        if (run_seed > 0) {
            radar_sim.setSeed(run_seed + 10000);
        }

        // Tracker（状態リセット）
        MultiTargetTracker tracker(max_tracks, ukf_params, assoc_params, process_noise, radar_params.meas_noise);
        tracker.setSensorPosition(sensor_x, sensor_y);

        // IMM遷移確率行列の設定（4×4: CLI上書きがあれば適用）
        std::vector<std::vector<float>> imm_matrix = {
            {cli_imm_ca_ca >= 0 ? cli_imm_ca_ca : 0.75f,
             cli_imm_ca_bal >= 0 ? cli_imm_ca_bal : 0.10f,
             cli_imm_ca_ct >= 0 ? cli_imm_ca_ct : 0.05f,
             cli_imm_ca_sg >= 0 ? cli_imm_ca_sg : 0.10f},
            {cli_imm_bal_ca >= 0 ? cli_imm_bal_ca : 0.08f,
             cli_imm_bal_bal >= 0 ? cli_imm_bal_bal : 0.72f,
             cli_imm_bal_ct >= 0 ? cli_imm_bal_ct : 0.05f,
             cli_imm_bal_sg >= 0 ? cli_imm_bal_sg : 0.15f},
            {cli_imm_ct_ca >= 0 ? cli_imm_ct_ca : 0.05f,
             cli_imm_ct_bal >= 0 ? cli_imm_ct_bal : 0.05f,
             cli_imm_ct_ct >= 0 ? cli_imm_ct_ct : 0.80f,
             cli_imm_ct_sg >= 0 ? cli_imm_ct_sg : 0.10f},
            {cli_imm_sg_ca >= 0 ? cli_imm_sg_ca : 0.10f,
             cli_imm_sg_bal >= 0 ? cli_imm_sg_bal : 0.15f,
             cli_imm_sg_ct >= 0 ? cli_imm_sg_ct : 0.05f,
             cli_imm_sg_sg >= 0 ? cli_imm_sg_sg : 0.70f}
        };
        tracker.setIMMTransitionMatrix(imm_matrix);

        // IMMモデルノイズ倍率の設定（CLI上書きがあれば適用）
        float ca_noise_mult = (cli_imm_ca_noise >= 0) ? cli_imm_ca_noise : 0.1f;
        float bal_noise_mult = (cli_imm_bal_noise >= 0) ? cli_imm_bal_noise : 0.3f;
        float ct_noise_mult = (cli_imm_ct_noise >= 0) ? cli_imm_ct_noise : 2.5f;
        float sg_noise_mult = (cli_imm_sg_noise >= 0) ? cli_imm_sg_noise : 1.5f;
        tracker.setIMMNoiseMultipliers(ca_noise_mult, bal_noise_mult, ct_noise_mult, sg_noise_mult);

        // Evaluator
        TrackingEvaluator evaluator(ospa_cutoff, 2);

        if (num_runs > 1) {
            std::cout << "\n--- Run " << (run + 1) << "/" << num_runs << " ---" << std::endl;
        } else {
            std::cout << "\nStarting simulation..." << std::endl;
        }
        std::cout << "Total frames: " << num_frames << std::endl;

        // CSVファイル出力（最終runのみ、または単一run時）
        bool write_csv = (run == num_runs - 1);

        std::ofstream out_file, track_file, ground_truth_file, meas_file;
        if (write_csv) {
            out_file.open(output_file);
            out_file << "frame,time,num_tracks,num_confirmed,num_measurements,processing_time_ms,beam_track,beam_search,beam_demand" << std::endl;

            track_file.open("track_details.csv");
            track_file << "frame,time,track_id,x,y,z,vx,vy,vz,ax,ay,az,state,model_prob_ca,model_prob_ballistic,model_prob_ct,model_prob_sg,misses,miss_reason" << std::endl;

            ground_truth_file.open("ground_truth.csv");
            ground_truth_file << "frame,time,target_id,x,y,z,vx,vy,vz,ax,ay,az" << std::endl;

            meas_file.open("measurements.csv");
            meas_file << "frame,time,range,azimuth,elevation,doppler,snr,true_target_id,beam_type" << std::endl;
        }

        double total_predict_ms = 0, total_assoc_ms = 0, total_update_ms = 0, total_frame_ms = 0;
        double max_frame_ms = 0, min_frame_ms = 1e9;
        auto wall_start = std::chrono::high_resolution_clock::now();

        // ビームステアリング: サーチ走査角（フレーム間で持続）
        float search_scan_angle = search_min_az;

        for (int frame = 0; frame < num_frames; frame++) {
            double current_time = frame * dt;

            auto ground_truth = radar_sim.getTrueStates(current_time);
            auto active_ids = target_gen.getActiveTargets(current_time);

            if (write_csv) {
                for (size_t i = 0; i < ground_truth.size(); i++) {
                    const auto& gt = ground_truth[i];
                    int target_id = (i < active_ids.size()) ? active_ids[i] : static_cast<int>(i);
                    ground_truth_file << frame << ","
                                    << current_time << ","
                                    << target_id << ","
                                    << gt(0) << "," << gt(1) << "," << gt(2) << ","
                                    << gt(3) << "," << gt(4) << "," << gt(5) << ","
                                    << gt(6) << "," << gt(7) << "," << gt(8) << std::endl;
                }
            }

            // ビームステアリング: ビーム方向計算
            // miss_reason: 0=探知, 1=覆域外, 2=ビームリソース不足, 3=ビーム照射したが未探知
            std::unordered_map<int, int> track_miss_reason;
            int beam_track_count = 0;   // 追尾ビーム数
            int beam_search_count = 0;  // サーチビーム数
            int beam_demand = 0;        // レーダー覆域内トラック数（ビーム要求数）
            {
                std::vector<float> beam_dirs;
                std::vector<float> beam_elevs;
                std::vector<float> beam_target_ranges;  // 追尾ビーム目標距離 (0=サーチ)
                float half_azimuth_cov = radar_params.azimuth_coverage / 2.0f;

                // 追尾ビーム覆域チェック用ラムダ: レーダー基本範囲＋仰角のみ（方位角制限なし）
                // track_range_width: 各目標を中心とした距離幅（将来的に目標ごとの範囲ゲート制御に使用可能）
                auto isInRadarCov = [&](float az, float el, float range) -> bool {
                    // 基本的なレーダー距離範囲チェック
                    if (range < radar_params.min_range || range > radar_params.max_range)
                        return false;
                    // 仰角チェック
                    if (el < radar_params.min_elevation || el > radar_params.max_elevation)
                        return false;
                    return true;  // NO azimuth check for track beams
                };

                // トラック選択: confirmed_onlyかall tracksか
                // トラック選択: confirmed_onlyかall tracksか
                // ブートストラップモード: CONFIRMED航跡がゼロの場合、TENTATIVE
                // 航跡にビームを割当てて早期確認を促進する。
                // 制約: (1) hits>=1, (2) サーチレンジ内, (3) 最大4航跡
                // hits>=1で十分: サーチビーム検出はSNR閾値を超えた真の検出であり、
                // クラッタ起源でもトラックビームで即座に棄却される。
                auto beam_tracks = track_confirmed_only
                    ? tracker.getConfirmedTracks()
                    : tracker.getAllTracks();
                // 拡張ブートストラップ: CONFIRMED航跡の有無に関わらず、
                // 全CONFIRMEDから1km以上離れたTENTATIVEにもビーム割当。
                // 分離後の新目標を迅速に獲得するために不可欠。
                // CONFIRMED=0の場合は従来通り全TENTATIVEが対象。
                if (track_confirmed_only) {
                    auto all = tracker.getAllTracks();
                    std::sort(all.begin(), all.end(),
                        [](const Track& a, const Track& b) { return a.hits > b.hits; });
                    int bootstrap_count = 0;
                    float min_dist_from_confirmed_sq = 2000.0f * 2000.0f; // 2km
                    for (const auto& t : all) {
                        if (bootstrap_count >= 4) break;
                        if (t.track_state != TrackState::TENTATIVE || t.hits < 1) continue;
                        // サーチレンジ内チェック
                        float bdx = t.state(0) - sensor_x;
                        float bdy = t.state(1) - sensor_y;
                        float bdz = t.state(2) - radar_params.sensor_z;
                        float range_3d = std::sqrt(bdx*bdx + bdy*bdy + bdz*bdz);
                        if (range_3d < search_min_range || range_3d > search_max_range) continue;
                        // CONFIRMED航跡がある場合: 全CONFIRMEDから1km以上離れていること
                        // CONFIRMED=0の場合: 条件なし（従来の挙動）
                        if (!beam_tracks.empty()) {
                            bool too_close = false;
                            for (const auto& ct : beam_tracks) {
                                float dx = t.state(0) - ct.state(0);
                                float dy = t.state(1) - ct.state(1);
                                float dz = t.state(2) - ct.state(2);
                                if (dx*dx + dy*dy + dz*dz < min_dist_from_confirmed_sq) {
                                    too_close = true;
                                    break;
                                }
                            }
                            if (too_close) continue;
                        }
                        beam_tracks.push_back(t);
                        bootstrap_count++;
                    }
                }

                // レーダー覆域内のトラックのみ抽出（ID追跡付き）
                std::vector<float> track_azimuths;
                std::vector<float> track_elevations;
                std::vector<float> track_ranges;  // 各トラックの3D距離
                std::vector<int> radar_cov_track_ids;
                for (const auto& t : beam_tracks) {
                    float bdx = t.state(0) - sensor_x;
                    float bdy = t.state(1) - sensor_y;
                    float bdz = t.state(2) - radar_params.sensor_z;
                    float r_horiz_t = std::sqrt(bdx * bdx + bdy * bdy);
                    float range_3d = std::sqrt(bdx * bdx + bdy * bdy + bdz * bdz);
                    float az = std::atan2(bdy, bdx);
                    float el = std::atan2(bdz, r_horiz_t);

                    // トラックビームはレーダー覆域内なら割当可能（距離＋仰角のみチェック、方位角制限なし）
                    // サーチ領域はサーチビームの配置範囲を定義するのみ
                    if (isInRadarCov(az, el, range_3d)) {
                        // ビーム剥奪: 連続ミスが多いCONFIRMED航跡は予測位置が
                        // 大幅に発散している可能性が高い（HGV機動時の高度誤差等）。
                        // 追尾ビームを無駄に消費するより、サーチビームに転用して
                        // 目標再検出確率を向上させる。
                        if (t.track_state == TrackState::CONFIRMED && t.misses >= 10) {
                            track_miss_reason[t.id] = 3;  // beam_miss（FOV内だがビーム剥奪）
                            continue;  // ビーム割当対象から除外→サーチビームに転用
                        }
                        track_azimuths.push_back(az);
                        track_elevations.push_back(el);
                        track_ranges.push_back(range_3d);
                        radar_cov_track_ids.push_back(t.id);
                        track_miss_reason[t.id] = 2;  // デフォルト: ビームリソース不足
                    } else {
                        track_miss_reason[t.id] = 1;  // 覆域外（レーダー距離範囲外または仰角範囲外）
                    }
                }

                int n_fov_tracks = static_cast<int>(track_azimuths.size());
                int max_track_beams = std::max(0, radar_params.num_beams - min_search_beams);
                int track_budget = std::min(n_fov_tracks, max_track_beams);

                // ランタイム警告: トラックあるがビームなし
                static bool warning_shown = false;
                if (!warning_shown && n_fov_tracks > 0 && max_track_beams == 0) {
                    std::cerr << "\n[Frame " << frame << " WARNING] "
                              << n_fov_tracks << " tracks need beams, but max_track_beams=0 "
                              << "(num_beams=" << radar_params.num_beams
                              << ", min_search_beams=" << min_search_beams << ")" << std::endl;
                    warning_shown = true;
                }

                // === CONFIRMED優先ビーム割当 ===
                // 1. CONFIRMED航跡に優先的にビームを割当
                // 2. 残りビームをTENTATIVE/LOST航跡にラウンドロビン割当
                {
                    // CONFIRMED航跡IDのセットを構築（O(1)検索）
                    std::unordered_set<int> confirmed_ids;
                    for (const auto& t : beam_tracks) {
                        if (t.track_state == TrackState::CONFIRMED) {
                            confirmed_ids.insert(t.id);
                        }
                    }

                    // CONFIRMED vs その他をインデックス分類
                    std::vector<int> confirmed_indices;
                    std::vector<int> other_indices;
                    for (int b = 0; b < n_fov_tracks; b++) {
                        if (confirmed_ids.count(radar_cov_track_ids[b])) {
                            confirmed_indices.push_back(b);
                        } else {
                            other_indices.push_back(b);
                        }
                    }

                    int beams_used = 0;
                    // Phase 1: 各CONFIRMED航跡に1ビームずつ割当
                    for (int ci : confirmed_indices) {
                        if (beams_used >= track_budget) break;
                        beam_dirs.push_back(track_azimuths[ci]);
                        beam_elevs.push_back(track_elevations[ci]);
                        beam_target_ranges.push_back(track_ranges[ci]);
                        track_miss_reason[radar_cov_track_ids[ci]] = 3;
                        beams_used++;
                    }

                    // Phase 2: 残りビームをTENTATIVE/LOST航跡にラウンドロビン割当
                    if (!other_indices.empty() && beams_used < track_budget) {
                        int remaining = track_budget - beams_used;
                        int offset = frame % static_cast<int>(other_indices.size());
                        for (int b = 0; b < remaining && b < static_cast<int>(other_indices.size()); b++) {
                            int idx = other_indices[(offset + b) % other_indices.size()];
                            beam_dirs.push_back(track_azimuths[idx]);
                            beam_elevs.push_back(track_elevations[idx]);
                            beam_target_ranges.push_back(track_ranges[idx]);
                            track_miss_reason[radar_cov_track_ids[idx]] = 3;
                            beams_used++;
                        }
                    }
                }

                // サーチビーム: サーチセクタ内を多仰角ラスタースキャン
                // 仰角をフレームごとに切替え、広い仰角範囲の目標を検出可能にする
                int search_count = radar_params.num_beams - static_cast<int>(beam_dirs.size());

                // 仰角スキャン範囲: CLI引数から設定（デフォルト: 0° ～ 20°）
                float elev_scan_min = elev_scan_min_arg;
                float elev_scan_max = elev_scan_max_arg;
                int num_elev_bars = elev_bars_per_frame_arg;
                int total_elev_positions = elev_cycle_steps_arg;
                // 実際に到達する最大インデックスを計算: floor((steps-1)/bars) + (bars-1)
                // 実際に到達する最大インデックス (modulo total_elev_positions で制限)
                int raw_max_idx = (total_elev_positions - 1) / num_elev_bars + (num_elev_bars - 1);
                int max_reachable_idx = std::min(raw_max_idx, total_elev_positions - 1);
                float elev_bar_step = (elev_scan_max <= elev_scan_min) ? 0.0f
                    : (elev_scan_max - elev_scan_min) / static_cast<float>(std::max(max_reachable_idx, 1));

                if (frame == 0) {
                    std::cout << "[ElevScan] min=" << (elev_scan_min * 180.0f / M_PI) << "° max=" << (elev_scan_max * 180.0f / M_PI)
                              << "° bars=" << num_elev_bars << " steps=" << total_elev_positions
                              << " max_idx=" << max_reachable_idx
                              << " step_size=" << (elev_bar_step * 180.0f / M_PI) << "°" << std::endl;
                }

                // フレームごとに仰角バーを巡回（3バー×3段階=9パターンで全仰角をカバー）
                int elev_cycle = frame % total_elev_positions;

                int beams_per_bar = std::max(1, search_count / num_elev_bars);
                int remaining_beams = search_count;

                // 仰角バー: 固定スキャンパターン
                for (int bar = 0; bar < num_elev_bars && remaining_beams > 0; bar++) {
                    // このバーの仰角: サイクル位置に応じてオフセット
                    int elev_idx = (elev_cycle / num_elev_bars + bar) % total_elev_positions;
                    float bar_elevation = elev_scan_min + elev_bar_step * static_cast<float>(elev_idx);
                    bar_elevation = std::max(elev_scan_min, std::min(bar_elevation, elev_scan_max));

                    int bar_beams = (bar == num_elev_bars - 1) ? remaining_beams : beams_per_bar;
                    for (int s = 0; s < bar_beams; s++) {
                        beam_dirs.push_back(search_scan_angle);
                        beam_elevs.push_back(bar_elevation);
                        beam_target_ranges.push_back(0.0f);
                        search_scan_angle += radar_params.beam_width;
                        if (search_scan_angle > search_max_az) {
                            search_scan_angle = search_min_az;
                        }
                    }
                    remaining_beams -= bar_beams;
                }

                radar_sim.setBeamDirections(beam_dirs);
                radar_sim.setBeamElevations(beam_elevs);
                radar_sim.setBeamTargetRanges(beam_target_ranges);

                // ビーム統計を外部変数に保存
                beam_demand = n_fov_tracks;
                beam_track_count = track_budget;
                beam_search_count = search_count;
            }

            auto measurements = radar_sim.generate(current_time);
            // Ground Truth 関連付けを取得（評価専用）
            const auto& true_associations = radar_sim.getTrueAssociations();
            const auto& beam_types = radar_sim.getBeamTypes();

            if (write_csv) {
                for (size_t i = 0; i < measurements.size(); i++) {
                    const auto& m = measurements[i];
                    int true_target_id = (i < true_associations.size()) ? true_associations[i] : -1;
                    int beam_type = (i < beam_types.size()) ? beam_types[i] : 0;
                    meas_file << frame << ","
                              << current_time << ","
                              << m.range << "," << m.azimuth << ","
                              << m.elevation << "," << m.doppler << ","
                              << m.snr << "," << true_target_id << ","
                              << beam_type << std::endl;
                }
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

            // 処理時間蓄積
            {
                const auto& pf = tracker.getLastPerformanceStats();
                total_predict_ms += pf.predict_time_ms;
                total_assoc_ms += pf.association_time_ms;
                total_update_ms += pf.update_time_ms;
                total_frame_ms += pf.total_time_ms;
                if (pf.total_time_ms > max_frame_ms) max_frame_ms = pf.total_time_ms;
                if (pf.total_time_ms < min_frame_ms) min_frame_ms = pf.total_time_ms;
            }

            auto all_tracks = tracker.getAllTracks();

            // Confirmed trackのみを抽出（内部IDをそのまま使用）
            std::vector<Track> confirmed_tracks;
            for (const auto& track : all_tracks) {
                if (track.track_state == TrackState::CONFIRMED) {
                    confirmed_tracks.push_back(track);
                }
            }

            // Evaluationはconfirmed trackのみで実施
            evaluator.update(confirmed_tracks, ground_truth, active_ids,
                            static_cast<int>(measurements.size()),
                            current_time);

            if (write_csv) {
                // CSVにはconfirmed trackのみ出力（内部ID使用）
                for (const auto& track : confirmed_tracks) {
                    int state_value = 1;  // All are CONFIRMED

                    float prob_cv = track.model_probs.size() >= 1 ? track.model_probs[0] : 0.25f;
                    float prob_high = track.model_probs.size() >= 2 ? track.model_probs[1] : 0.25f;
                    float prob_med = track.model_probs.size() >= 3 ? track.model_probs[2] : 0.25f;
                    float prob_sg = track.model_probs.size() >= 4 ? track.model_probs[3] : 0.25f;

                    // miss_reason: 0=探知, 1=覆域外, 2=ビームリソース不足, 3=ビーム照射未探知
                    int reason = 0;
                    if (track.misses > 0) {
                        auto it = track_miss_reason.find(track.id);
                        reason = (it != track_miss_reason.end()) ? it->second : 3;
                    }

                    track_file << frame << "," << current_time << ","
                              << track.id << ","  // 連番ID
                              << track.state(0) << "," << track.state(1) << "," << track.state(2) << ","
                              << track.state(3) << "," << track.state(4) << "," << track.state(5) << ","
                              << track.state(6) << "," << track.state(7) << "," << track.state(8) << ","
                              << state_value << ","
                              << prob_cv << "," << prob_high << "," << prob_med << "," << prob_sg << ","
                              << track.misses << "," << reason << std::endl;
                }

                const auto& perf = tracker.getLastPerformanceStats();
                out_file << frame << "," << current_time << ","
                         << tracker.getNumTracks() << ","
                         << tracker.getNumConfirmedTracks() << ","
                         << measurements.size() << ","
                         << perf.total_time_ms << ","
                         << beam_track_count << ","
                         << beam_search_count << ","
                         << beam_demand << std::endl;
            }

            if ((frame + 1) % 10 == 0 || frame == 0) {
                const auto& perf = tracker.getLastPerformanceStats();
                std::cout << "Frame " << (frame + 1) << "/" << num_frames
                          << " | Tracks: " << tracker.getNumConfirmedTracks()
                          << " | Meas: " << measurements.size()
                          << " | Time: " << std::fixed << std::setprecision(2)
                          << perf.total_time_ms << " ms" << std::endl;
            }
        }

        if (write_csv) {
            out_file.close();
            track_file.close();
            ground_truth_file.close();
            meas_file.close();
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

        std::cout << "\n=== Simulation Complete ===" << std::endl;

        // Timing summary
        double avg_frame_ms = (num_frames > 0) ? total_frame_ms / num_frames : 0;
        std::cout << "\n[Performance]" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Wall-clock time: " << wall_ms << " ms (" << (wall_ms / 1000.0) << " s)" << std::endl;
        std::cout << "  GPU total: " << total_frame_ms << " ms" << std::endl;
        std::cout << "  GPU avg/frame: " << avg_frame_ms << " ms" << std::endl;
        std::cout << "  GPU min/frame: " << min_frame_ms << " ms" << std::endl;
        std::cout << "  GPU max/frame: " << max_frame_ms << " ms" << std::endl;
        std::cout << "  Predict total: " << total_predict_ms << " ms" << std::endl;
        std::cout << "  Association total: " << total_assoc_ms << " ms" << std::endl;
        std::cout << "  Update total: " << total_update_ms << " ms" << std::endl;
        if (wall_ms > 0) {
            std::cout << "  Realtime factor: " << std::setprecision(1) << (duration * 1000.0 / wall_ms) << "x" << std::endl;
        }

        tracker.printStatistics();
        radar_sim.printStatistics();
        evaluator.printSummary();

        if (write_csv) {
            evaluator.exportToCSV("evaluation_results.csv");
        }

        // Accumulate metrics
        all_accuracy[run] = evaluator.computeAccuracyMetrics();
        all_detection[run] = evaluator.computeDetectionMetrics();
        all_wall_time_ms[run] = wall_ms;
        all_avg_frame_ms[run] = avg_frame_ms;
    }

    // Aggregated output for multi-run
    if (num_runs > 1) {
        // Statistics helper
        auto mean_std = [](const std::vector<float>& v) -> std::pair<float, float> {
            float sum = 0.0f;
            for (float x : v) sum += x;
            float mean = sum / static_cast<float>(v.size());
            float sq_sum = 0.0f;
            for (float x : v) sq_sum += (x - mean) * (x - mean);
            float stddev = std::sqrt(sq_sum / static_cast<float>(v.size()));
            return {mean, stddev};
        };

        std::vector<float> pos_rmse(num_runs), vel_rmse(num_runs), ospa(num_runs);
        std::vector<float> recall(num_runs), f1(num_runs), precision(num_runs);
        for (int i = 0; i < num_runs; i++) {
            pos_rmse[i] = all_accuracy[i].position_rmse;
            vel_rmse[i] = all_accuracy[i].velocity_rmse;
            ospa[i] = all_accuracy[i].ospa_distance;
            recall[i] = all_detection[i].recall * 100.0f;
            f1[i] = all_detection[i].f1_score * 100.0f;
            precision[i] = all_detection[i].precision * 100.0f;
        }

        auto [pos_mean, pos_std] = mean_std(pos_rmse);
        auto [vel_mean, vel_std] = mean_std(vel_rmse);
        auto [ospa_mean, ospa_std] = mean_std(ospa);
        auto [recall_mean, recall_std] = mean_std(recall);
        auto [f1_mean, f1_std] = mean_std(f1);
        auto [prec_mean, prec_std] = mean_std(precision);

        std::cout << "\n========================================" << std::endl;
        std::cout << "=== Aggregated Results (" << num_runs << " runs) ===" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Position RMSE: " << pos_mean << " +/- " << pos_std << " m" << std::endl;
        std::cout << "  Velocity RMSE: " << vel_mean << " +/- " << vel_std << " m/s" << std::endl;
        std::cout << "  OSPA Distance: " << ospa_mean << " +/- " << ospa_std << " m" << std::endl;
        std::cout << "  Precision: " << prec_mean << " +/- " << prec_std << " %" << std::endl;
        std::cout << "  Recall: " << recall_mean << " +/- " << recall_std << " %" << std::endl;
        std::cout << "  F1 Score: " << f1_mean << " +/- " << f1_std << " %" << std::endl;

        // Timing aggregation
        std::vector<float> wall_times(num_runs), avg_frames(num_runs);
        for (int i = 0; i < num_runs; i++) {
            wall_times[i] = static_cast<float>(all_wall_time_ms[i]);
            avg_frames[i] = static_cast<float>(all_avg_frame_ms[i]);
        }
        auto [wall_mean, wall_std] = mean_std(wall_times);
        auto [frame_mean, frame_std] = mean_std(avg_frames);
        std::cout << "  Wall-clock: " << (wall_mean / 1000.0f) << " +/- " << (wall_std / 1000.0f) << " s" << std::endl;
        std::cout << "  Avg frame: " << frame_mean << " +/- " << frame_std << " ms" << std::endl;
        std::cout << "========================================" << std::endl;
    }

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
    float fuel_fraction = 0.70f;
    float specific_impulse = 250.0f;
    float drag_coefficient = 0.3f;
    float cross_section_area = 1.0f;

    // センサーパラメータ
    float sensor_x = 0.0f, sensor_y = 0.0f;
    float radar_min_range = 0.0f;
    float radar_max_range = 0.0f;
    float azimuth_coverage = 2.0f * static_cast<float>(M_PI);  // 方位覆域幅 [rad]
    float min_elevation = -0.5236f;  // 最小仰角 [rad] (-30°)
    float max_elevation = 1.5708f;   // 最大仰角 [rad] (+90°)
    float radar_fov = 2.0f * static_cast<float>(M_PI);  // DEPRECATED: 後方互換性のため保持

    // 座標変換用
    double target_lat = 35.6762, target_lon = 139.6503;

    // 自動調整用
    float distance_threshold = 10000.0f;  // 10km デフォルト
    float target_max_altitude = 0.0f;     // 目標最高高度 [m]（0=制約なし）
    bool lock_angle = false;
    bool lock_isp = false;
    bool lock_fuel = false;

    // ミサイルタイプ (ballistic or hgv)
    std::string missile_type = "ballistic";

    // HGV固有パラメータ
    float cruise_altitude = 40000.0f;     // 巡航高度 [m]
    float glide_ratio = 4.0f;            // 揚抗比 (L/D)
    float terminal_dive_range = 20000.0f; // 終末ダイブ開始距離 [m]
    float pullup_duration = 30.0f;        // 引き起こし遷移時間 [s]
    float bank_angle_max = 0.0f;          // 最大バンク角 [rad]
    int num_skips = 1;                    // スキップ回数 (0=無制限)

    // 分離シミュレーション用
    bool enable_separation = false;
    float warhead_mass_fraction = 0.3f;
    float warhead_cd = 0.15f;
    float booster_cd = 1.5f;

    // 複数回実行用
    int num_runs = 1;
    uint32_t seed = 0;

    // クラスタ目標
    int cluster_count = 0;
    float cluster_spread = 5000.0f;  // [m]
    float launch_time_spread = 5.0f; // [s] 発射時刻のばらつき

    // ビームステアリング
    float beam_width = 0.052f;   // ~3 degrees
    int num_beams = 10;
    int min_search_beams = 1;
    bool track_confirmed_only = false;
    float search_sector = -1.0f;  // サーチセクタ幅 [rad] (-1 = FOV全体)
    float search_center = 0.0f;   // サーチセクタ中心方位 [rad] (atan2系)
    float antenna_boresight = 0.0f; // アンテナ中心方位 [rad] (atan2系)
    float search_min_range = 0.0f;  // サーチ領域最小距離 [m]
    float search_max_range = 0.0f;  // サーチ領域最大距離 [m] (0 = レーダー最大距離と同じ)
    float track_range_width = 0.0f; // 追尾ビーム距離幅 [m] (目標距離±width/2, 0 = 制限なし)
    float range_resolution = 150.0f; // レンジ分解能 [m] (0 = 分解能制限なし)

    // 多仰角サーチスキャンパラメータ
    float elev_scan_min = 0.0f;       // 仰角スキャン下限 [rad]
    float elev_scan_max = 0.35f;      // 仰角スキャン上限 [rad] (~20°)
    int elev_bars_per_frame = 3;      // フレームあたりの仰角バー数
    int elev_cycle_steps = 9;         // 仰角巡回ステップ数

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
        else if (arg == "--radar-min-range" && i + 1 < argc) radar_min_range = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--radar-max-range" && i + 1 < argc) radar_max_range = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--azimuth-coverage" && i + 1 < argc) azimuth_coverage = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--min-elevation" && i + 1 < argc) min_elevation = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--max-elevation" && i + 1 < argc) max_elevation = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--radar-fov" && i + 1 < argc) {
            azimuth_coverage = static_cast<float>(std::atof(argv[++i]));
            std::cerr << "Warning: --radar-fov is deprecated, use --azimuth-coverage instead" << std::endl;
        }
        else if (arg == "--distance-threshold" && i + 1 < argc) distance_threshold = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--target-max-altitude" && i + 1 < argc) target_max_altitude = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--lock-angle") lock_angle = true;
        else if (arg == "--lock-isp") lock_isp = true;
        else if (arg == "--lock-fuel") lock_fuel = true;
        else if (arg == "--missile-type" && i + 1 < argc) missile_type = argv[++i];
        else if (arg == "--cruise-altitude" && i + 1 < argc) cruise_altitude = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glide-ratio" && i + 1 < argc) glide_ratio = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--terminal-dive-range" && i + 1 < argc) terminal_dive_range = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pullup-duration" && i + 1 < argc) pullup_duration = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--bank-angle-max" && i + 1 < argc) bank_angle_max = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--num-skips" && i + 1 < argc) num_skips = std::atoi(argv[++i]);
        else if (arg == "--enable-separation") enable_separation = true;
        else if (arg == "--warhead-mass-fraction" && i + 1 < argc) warhead_mass_fraction = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--warhead-cd" && i + 1 < argc) warhead_cd = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--booster-cd" && i + 1 < argc) booster_cd = static_cast<float>(std::atof(argv[++i]));
        // トラッカーパラメータ
        else if (arg == "--gate-threshold" && i + 1 < argc) cli_gate_threshold = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--confirm-hits" && i + 1 < argc) cli_confirm_hits = std::atoi(argv[++i]);
        else if (arg == "--confirm-window" && i + 1 < argc) cli_confirm_window = std::atoi(argv[++i]);
        else if (arg == "--delete-misses" && i + 1 < argc) cli_delete_misses = std::atoi(argv[++i]);
        else if (arg == "--min-snr" && i + 1 < argc) cli_min_snr = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--max-jump-velocity" && i + 1 < argc) cli_max_jump_velocity = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--min-init-distance" && i + 1 < argc) cli_min_init_distance = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--process-pos-noise" && i + 1 < argc) cli_process_pos = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--process-vel-noise" && i + 1 < argc) cli_process_vel = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--process-acc-noise" && i + 1 < argc) cli_process_acc = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--detect-prob" && i + 1 < argc) cli_detect_prob = static_cast<float>(std::atof(argv[++i]));
        // JPDA パラメータ
        else if (arg == "--association" && i + 1 < argc) cli_association_method = argv[++i];
        else if (arg == "--jpda-pd" && i + 1 < argc) cli_jpda_pd = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--jpda-clutter-density" && i + 1 < argc) cli_jpda_clutter_density = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--jpda-gate" && i + 1 < argc) cli_jpda_gate = static_cast<float>(std::atof(argv[++i]));
        // PMBM パラメータ
        else if (arg == "--pmbm-pd" && i + 1 < argc) cli_pmbm_pd = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pmbm-gate" && i + 1 < argc) cli_pmbm_gate = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pmbm-clutter-density" && i + 1 < argc) cli_pmbm_clutter_density = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pmbm-k-best" && i + 1 < argc) cli_pmbm_k_best = std::atoi(argv[++i]);
        else if (arg == "--pmbm-survival" && i + 1 < argc) cli_pmbm_survival = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pmbm-init-existence" && i + 1 < argc) cli_pmbm_init_existence = static_cast<float>(std::atof(argv[++i]));
        // MHT パラメータ
        else if (arg == "--mht-pd" && i + 1 < argc) cli_mht_pd = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--mht-gate" && i + 1 < argc) cli_mht_gate = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--mht-clutter-density" && i + 1 < argc) cli_mht_clutter_density = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--mht-k-best" && i + 1 < argc) cli_mht_k_best = std::atoi(argv[++i]);
        else if (arg == "--mht-max-hypotheses" && i + 1 < argc) cli_mht_max_hypotheses = std::atoi(argv[++i]);
        else if (arg == "--mht-score-decay" && i + 1 < argc) cli_mht_score_decay = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--mht-prune-ratio" && i + 1 < argc) cli_mht_prune_ratio = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--mht-switch-cost" && i + 1 < argc) cli_mht_switch_cost = static_cast<float>(std::atof(argv[++i]));
        // GLMB パラメータ
        else if (arg == "--glmb-pd" && i + 1 < argc) cli_glmb_pd = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glmb-gate" && i + 1 < argc) cli_glmb_gate = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glmb-clutter-density" && i + 1 < argc) cli_glmb_clutter_density = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glmb-k-best" && i + 1 < argc) cli_glmb_k_best = std::atoi(argv[++i]);
        else if (arg == "--glmb-max-hypotheses" && i + 1 < argc) cli_glmb_max_hypotheses = std::atoi(argv[++i]);
        else if (arg == "--glmb-survival" && i + 1 < argc) cli_glmb_survival = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glmb-birth-weight" && i + 1 < argc) cli_glmb_birth_weight = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glmb-score-decay" && i + 1 < argc) cli_glmb_score_decay = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glmb-init-existence" && i + 1 < argc) cli_glmb_init_existence = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--glmb-sampler" && i + 1 < argc) cli_glmb_sampler = argv[++i];
        else if (arg == "--glmb-gibbs-sweeps" && i + 1 < argc) cli_glmb_gibbs_sweeps = std::atoi(argv[++i]);
        else if (arg == "--glmb-gibbs-burnin" && i + 1 < argc) cli_glmb_gibbs_burnin = std::atoi(argv[++i]);
        // UKF パラメータ
        else if (arg == "--ukf-alpha" && i + 1 < argc) cli_ukf_alpha = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--ukf-beta" && i + 1 < argc) cli_ukf_beta = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--ukf-kappa" && i + 1 < argc) cli_ukf_kappa = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--max-distance" && i + 1 < argc) cli_max_distance = static_cast<float>(std::atof(argv[++i]));
        // IMM 遷移確率 (4×4: CA, Ballistic, CT, SkipGlide)
        else if (arg == "--imm-ca-ca" && i + 1 < argc) cli_imm_ca_ca = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ca-bal" && i + 1 < argc) cli_imm_ca_bal = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ca-ct" && i + 1 < argc) cli_imm_ca_ct = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ca-sg" && i + 1 < argc) cli_imm_ca_sg = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-bal-ca" && i + 1 < argc) cli_imm_bal_ca = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-bal-bal" && i + 1 < argc) cli_imm_bal_bal = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-bal-ct" && i + 1 < argc) cli_imm_bal_ct = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-bal-sg" && i + 1 < argc) cli_imm_bal_sg = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ct-ca" && i + 1 < argc) cli_imm_ct_ca = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ct-bal" && i + 1 < argc) cli_imm_ct_bal = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ct-ct" && i + 1 < argc) cli_imm_ct_ct = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ct-sg" && i + 1 < argc) cli_imm_ct_sg = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-sg-ca" && i + 1 < argc) cli_imm_sg_ca = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-sg-bal" && i + 1 < argc) cli_imm_sg_bal = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-sg-ct" && i + 1 < argc) cli_imm_sg_ct = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-sg-sg" && i + 1 < argc) cli_imm_sg_sg = static_cast<float>(std::atof(argv[++i]));
        // IMM モデルノイズ倍率
        else if (arg == "--imm-ca-noise" && i + 1 < argc) cli_imm_ca_noise = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-bal-noise" && i + 1 < argc) cli_imm_bal_noise = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-ct-noise" && i + 1 < argc) cli_imm_ct_noise = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--imm-sg-noise" && i + 1 < argc) cli_imm_sg_noise = static_cast<float>(std::atof(argv[++i]));
        // センサーパラメータ
        else if (arg == "--range-noise" && i + 1 < argc) cli_range_noise = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--azimuth-noise" && i + 1 < argc) cli_azimuth_noise = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--elevation-noise" && i + 1 < argc) cli_elevation_noise = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--doppler-noise" && i + 1 < argc) cli_doppler_noise = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pfa" && i + 1 < argc) cli_pfa = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--snr-ref" && i + 1 < argc) cli_snr_ref = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pd-ref" && i + 1 < argc) cli_pd_ref = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--pd-ref-range" && i + 1 < argc) cli_pd_ref_range = static_cast<float>(std::atof(argv[++i]));
        // 複数回実行
        else if (arg == "--num-runs" && i + 1 < argc) num_runs = std::atoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint32_t>(std::atoi(argv[++i]));
        // クラスタ目標
        else if (arg == "--cluster-count" && i + 1 < argc) cluster_count = std::atoi(argv[++i]);
        else if (arg == "--cluster-spread" && i + 1 < argc) cluster_spread = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--launch-time-spread" && i + 1 < argc) launch_time_spread = static_cast<float>(std::atof(argv[++i]));
        // ビームステアリング
        else if (arg == "--beam-width" && i + 1 < argc) beam_width = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--num-beams" && i + 1 < argc) num_beams = std::atoi(argv[++i]);
        else if (arg == "--min-search-beams" && i + 1 < argc) min_search_beams = std::atoi(argv[++i]);
        else if (arg == "--track-confirmed-only") track_confirmed_only = true;
        else if (arg == "--search-sector" && i + 1 < argc) search_sector = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--search-center" && i + 1 < argc) search_center = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--antenna-boresight" && i + 1 < argc) antenna_boresight = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--search-elevation" && i + 1 < argc) { ++i; } // deprecated, ignored
        else if (arg == "--search-min-range" && i + 1 < argc) search_min_range = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--search-max-range" && i + 1 < argc) search_max_range = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--track-range-width" && i + 1 < argc) track_range_width = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--range-resolution" && i + 1 < argc) range_resolution = static_cast<float>(std::atof(argv[++i]));
        // 多仰角サーチスキャン
        else if (arg == "--elev-scan-min" && i + 1 < argc) elev_scan_min = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--elev-scan-max" && i + 1 < argc) elev_scan_max = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--elev-bars-per-frame" && i + 1 < argc) elev_bars_per_frame = std::atoi(argv[++i]);
        else if (arg == "--elev-cycle-steps" && i + 1 < argc) elev_cycle_steps = std::atoi(argv[++i]);
    }

    if (mode == "trajectory") {
        return runTrajectoryMode(
            launch_x, launch_y, target_x, target_y,
            launch_angle, boost_duration,
            initial_mass, fuel_fraction, specific_impulse,
            drag_coefficient, cross_section_area,
            target_lat, target_lon,
            1.0 / framerate,
            enable_separation, warhead_mass_fraction, warhead_cd, booster_cd,
            missile_type, cruise_altitude, glide_ratio,
            terminal_dive_range, pullup_duration, bank_angle_max,
            num_skips);
    }

    if (mode == "auto-adjust") {
        return runAutoAdjustMode(
            launch_x, launch_y, target_x, target_y,
            launch_angle, boost_duration,
            initial_mass, fuel_fraction, specific_impulse,
            drag_coefficient, cross_section_area,
            target_lat, target_lon,
            1.0 / framerate,
            distance_threshold,
            target_max_altitude,
            lock_angle, lock_isp, lock_fuel,
            enable_separation, warhead_mass_fraction, warhead_cd, booster_cd,
            missile_type, cruise_altitude, glide_ratio,
            terminal_dive_range, pullup_duration, bank_angle_max,
            num_skips);
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
        sensor_x, sensor_y, radar_min_range, radar_max_range, radar_fov,
        azimuth_coverage, min_elevation, max_elevation,
        enable_separation, warhead_mass_fraction, warhead_cd, booster_cd,
        num_runs, seed,
        missile_type, cruise_altitude, glide_ratio,
        terminal_dive_range, pullup_duration, bank_angle_max,
        num_skips,
        cluster_count, cluster_spread, launch_time_spread,
        beam_width, num_beams, min_search_beams, track_confirmed_only,
        search_sector, search_center, antenna_boresight,
        search_min_range, search_max_range, track_range_width, range_resolution,
        elev_scan_min, elev_scan_max, elev_bars_per_frame, elev_cycle_steps);
}
