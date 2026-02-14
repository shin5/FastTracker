#ifndef FASTTRACKER_TARGET_GENERATOR_HPP
#define FASTTRACKER_TARGET_GENERATOR_HPP

#include <vector>
#include <array>
#include <random>
#include "utils/types.hpp"

namespace fasttracker {

/**
 * @brief 目標軌跡のモーションモデル
 */
enum class MotionModel {
    CONSTANT_VELOCITY,      // 等速直線運動
    CONSTANT_ACCELERATION,  // 等加速度運動
    MANEUVERING,           // 機動目標（旋回、加減速）
    BALLISTIC_MISSILE,     // 弾道ミサイル
    HYPERSONIC_GLIDE       // 超音速滑空体（HGV）
};

/**
 * @brief 弾道ミサイルの飛翔フェーズ
 */
enum class BallisticPhase {
    BOOST,      // ブースト段階（上昇）
    MIDCOURSE,  // 中間飛翔段階（放物線）
    TERMINAL    // 終末段階（落下・機動）
};

/**
 * @brief RK4軌道キャッシュの1点
 */
struct TrajectoryPoint {
    double time;
    float x, y, z;       // 位置 [m] (z=高度)
    float vx, vy, vz;    // 速度 [m/s]
    float mass;           // 現在質量 [kg]
    BallisticPhase phase; // 飛翔フェーズ
};

/**
 * @brief ミサイルパラメータ
 */
struct MissileParameters {
    // 弾道ミサイル用
    float launch_angle;         // 発射角度 [rad]
    float max_altitude;         // 最大高度 [m]
    float boost_duration;       // ブースト時間 [s]
    float boost_acceleration;   // ブースト加速度 [m/s²]（レガシー互換用）
    Eigen::Vector2f target_position;  // 着弾目標位置 [x, y]

    // 物理モデルパラメータ
    float initial_mass;         // 初期質量 [kg]
    float fuel_fraction;        // 燃料質量割合
    float specific_impulse;     // 比推力 [s]
    float drag_coefficient;     // 抗力係数 Cd
    float cross_section_area;   // 参照断面積 [m²]

    // 超音速滑空体用
    float cruise_altitude;      // 巡航高度 [m]
    float glide_ratio;         // 滑空比（L/D）
    float skip_amplitude;      // スキップ振幅 [m]
    float skip_frequency;      // スキップ周波数 [Hz]
    float terminal_dive_range; // 終末ダイブ開始距離 [m]

    MissileParameters()
        : launch_angle(0.785f),         // 45度
          max_altitude(50000.0f),       // 50km
          boost_duration(60.0f),        // 60秒
          boost_acceleration(20.0f),    // 2G（レガシー）
          initial_mass(20000.0f),       // 20t
          fuel_fraction(0.65f),         // 65%燃料
          specific_impulse(250.0f),     // Isp 250s
          drag_coefficient(0.3f),       // Cd 0.3
          cross_section_area(1.0f),     // 1m²
          cruise_altitude(30000.0f),    // 30km
          glide_ratio(4.0f),            // L/D = 4
          skip_amplitude(5000.0f),      // 5km
          skip_frequency(0.01f),        // 0.01Hz
          terminal_dive_range(10000.0f) {  // 10km
        target_position.setZero();
    }
};

/**
 * @brief 目標軌跡生成パラメータ
 */
struct TargetParameters {
    StateVector initial_state;      // 初期状態
    MotionModel motion_model;       // モーションモデル
    float turn_rate;                // 旋回レート [rad/s]（機動目標用）
    float accel_magnitude;          // 加速度の大きさ [m/s²]（機動目標用）
    double birth_time;              // 出現時刻 [s]
    double death_time;              // 消失時刻 [s]（負の場合は無限）

    // ミサイル用パラメータ
    MissileParameters missile_params;

    // RK4軌道キャッシュ（物理ベース弾道用）
    mutable std::vector<TrajectoryPoint> trajectory_cache;
    mutable bool trajectory_computed = false;

    TargetParameters()
        : motion_model(MotionModel::CONSTANT_VELOCITY),
          turn_rate(0.0f),
          accel_magnitude(0.0f),
          birth_time(0.0),
          death_time(-1.0) {
        initial_state.setZero();
    }
};

/**
 * @brief 目標軌跡生成クラス
 */
class TargetGenerator {
public:
    TargetGenerator(int num_targets, float area_size = 10000.0f,
                    float max_velocity = 300.0f);

    void initializeDefaultScenario();
    void initializeCustomScenario(const std::vector<TargetParameters>& target_params);
    std::vector<StateVector> generateStates(double time) const;
    std::vector<int> getActiveTargets(double time) const;

    void generateClusteredScenario(const Eigen::Vector2f& cluster_center,
                                    float cluster_radius = 500.0f);
    void generateHighManeuverScenario();
    void generateBallisticMissileScenario(int num_missiles);
    void generateHypersonicGlideScenario(int num_hgvs);
    void generateMixedThreatScenario();

    int getNumTargets() const { return num_targets_; }

    const std::vector<TargetParameters>& getTargetParams() const {
        return target_params_;
    }

    /**
     * @brief 弾道軌道を事前計算（RK4物理積分）
     */
    void precomputeBallisticTrajectory(TargetParameters& params, bool verbose = true) const;

    /**
     * @brief 指定目標の最後に参照した高度を取得
     */
    float getLastAltitude(int target_idx) const;

    /**
     * @brief 弾道ミサイルの飛翔フェーズを判定
     */
    BallisticPhase getBallisticPhase(const TargetParameters& params, double time) const;

private:
    int num_targets_;
    float area_size_;
    float max_velocity_;
    std::vector<TargetParameters> target_params_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> uniform_dist_;
    mutable std::normal_distribution<float> normal_dist_;
    mutable std::vector<float> last_altitudes_;

    StateVector generateRandomInitialState();
    StateVector propagateConstantVelocity(const StateVector& state, double dt) const;
    StateVector propagateConstantAcceleration(const StateVector& state, double dt) const;
    StateVector propagateManeuver(const StateVector& state, double dt,
                                   float turn_rate, float accel_mag) const;
    StateVector propagateBallisticMissile(const TargetParameters& params, double time) const;
    StateVector propagateHypersonicGlide(const TargetParameters& params, double time) const;

    // 物理定数
    static constexpr float GRAVITY = 9.81f;
    static constexpr float EARTH_RADIUS = 6371000.0f;
    static constexpr float RHO0 = 1.225f;          // 海面大気密度 [kg/m³]
    static constexpr float SCALE_HEIGHT = 7400.0f;  // スケールハイト [m]
};

} // namespace fasttracker

#endif // FASTTRACKER_TARGET_GENERATOR_HPP
