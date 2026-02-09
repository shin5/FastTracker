#ifndef FASTTRACKER_TARGET_GENERATOR_HPP
#define FASTTRACKER_TARGET_GENERATOR_HPP

#include <vector>
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
 * @brief ミサイルパラメータ
 */
struct MissileParameters {
    // 弾道ミサイル用
    float launch_angle;         // 発射角度 [rad]
    float max_altitude;         // 最大高度 [m]
    float boost_duration;       // ブースト時間 [s]
    float boost_acceleration;   // ブースト加速度 [m/s²]
    Eigen::Vector2f target_position;  // 着弾目標位置 [x, y]

    // 超音速滑空体用
    float cruise_altitude;      // 巡航高度 [m]
    float glide_ratio;         // 滑空比（L/D）
    float skip_amplitude;      // スキップ振幅 [m]
    float skip_frequency;      // スキップ周波数 [Hz]
    float terminal_dive_range; // 終末ダイブ開始距離 [m]

    MissileParameters()
        : launch_angle(0.785f),      // 45度
          max_altitude(50000.0f),    // 50km
          boost_duration(60.0f),     // 60秒
          boost_acceleration(20.0f), // 2G
          cruise_altitude(30000.0f), // 30km
          glide_ratio(4.0f),         // L/D = 4
          skip_amplitude(5000.0f),   // 5km
          skip_frequency(0.01f),     // 0.01Hz
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
 *
 * 複数の目標の軌跡を生成します。
 * 各目標は異なるモーションモデルを持つことができます。
 */
class TargetGenerator {
public:
    /**
     * @brief コンストラクタ
     * @param num_targets 目標数
     * @param area_size 生成領域のサイズ [m]
     * @param max_velocity 最大速度 [m/s]
     */
    TargetGenerator(int num_targets, float area_size = 10000.0f,
                    float max_velocity = 300.0f);

    /**
     * @brief デフォルトシナリオで目標パラメータを初期化
     *
     * 目標を3つのグループに分けます：
     * - 70%: 等速直線運動
     * - 20%: 等加速度運動
     * - 10%: 機動目標
     */
    void initializeDefaultScenario();

    /**
     * @brief カスタムシナリオで初期化
     * @param target_params 各目標のパラメータ
     */
    void initializeCustomScenario(const std::vector<TargetParameters>& target_params);

    /**
     * @brief 指定時刻での目標状態を生成
     * @param time 時刻 [s]
     * @return 各目標の状態ベクトル
     */
    std::vector<StateVector> generateStates(double time) const;

    /**
     * @brief 指定時刻でのアクティブな目標のインデックスを取得
     * @param time 時刻 [s]
     * @return アクティブな目標のインデックスリスト
     */
    std::vector<int> getActiveTargets(double time) const;

    /**
     * @brief 密集シナリオを生成（データアソシエーションテスト用）
     * @param cluster_center クラスタ中心位置 [x, y]
     * @param cluster_radius クラスタ半径 [m]
     */
    void generateClusteredScenario(const Eigen::Vector2f& cluster_center,
                                    float cluster_radius = 500.0f);

    /**
     * @brief 高機動目標シナリオを生成（UKF性能テスト用）
     */
    void generateHighManeuverScenario();

    /**
     * @brief 弾道ミサイルシナリオを生成
     * @param num_missiles ミサイル数
     */
    void generateBallisticMissileScenario(int num_missiles);

    /**
     * @brief 超音速滑空体シナリオを生成
     * @param num_hgvs HGV数
     */
    void generateHypersonicGlideScenario(int num_hgvs);

    /**
     * @brief 複合脅威シナリオを生成（航空機＋ミサイル）
     */
    void generateMixedThreatScenario();

    /**
     * @brief 目標数を取得
     */
    int getNumTargets() const { return num_targets_; }

    /**
     * @brief 目標パラメータを取得
     */
    const std::vector<TargetParameters>& getTargetParams() const {
        return target_params_;
    }

private:
    int num_targets_;
    float area_size_;
    float max_velocity_;
    std::vector<TargetParameters> target_params_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> uniform_dist_;
    mutable std::normal_distribution<float> normal_dist_;

    /**
     * @brief ランダムな初期状態を生成
     */
    StateVector generateRandomInitialState();

    /**
     * @brief 等速直線運動モデルで状態を更新
     */
    StateVector propagateConstantVelocity(const StateVector& state, double dt) const;

    /**
     * @brief 等加速度運動モデルで状態を更新
     */
    StateVector propagateConstantAcceleration(const StateVector& state, double dt) const;

    /**
     * @brief 機動目標モデルで状態を更新
     */
    StateVector propagateManeuver(const StateVector& state, double dt,
                                   float turn_rate, float accel_mag) const;

    /**
     * @brief 弾道ミサイル軌道を計算
     */
    StateVector propagateBallisticMissile(const TargetParameters& params, double time) const;

    /**
     * @brief 超音速滑空体軌道を計算
     */
    StateVector propagateHypersonicGlide(const TargetParameters& params, double time) const;

    /**
     * @brief 弾道ミサイルの飛翔フェーズを判定
     */
    BallisticPhase getBallisticPhase(const TargetParameters& params, double time) const;

    /**
     * @brief 重力加速度 [m/s²]
     */
    static constexpr float GRAVITY = 9.81f;
};

} // namespace fasttracker

#endif // FASTTRACKER_TARGET_GENERATOR_HPP
