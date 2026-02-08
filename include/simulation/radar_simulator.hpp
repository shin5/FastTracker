#ifndef FASTTRACKER_RADAR_SIMULATOR_HPP
#define FASTTRACKER_RADAR_SIMULATOR_HPP

#include <vector>
#include <random>
#include "utils/types.hpp"
#include "target_generator.hpp"

namespace fasttracker {

/**
 * @brief レーダーパラメータ
 */
struct RadarParameters {
    float detection_probability;    // 検出確率 Pd
    float false_alarm_rate;         // 誤警報率（クラッタ密度） [1/m²]
    float max_range;                // 最大探知距離 [m]
    float field_of_view;            // 視野角 [rad]
    MeasurementNoise meas_noise;    // 観測ノイズ

    RadarParameters()
        : detection_probability(0.95f),
          false_alarm_rate(1e-8f),      // より現実的なクラッタ密度
          max_range(20000.0f),            // 20km監視範囲
          field_of_view(M_PI) {}  // 180度
};

/**
 * @brief レーダー観測シミュレータ
 *
 * 目標からの観測とクラッタを生成します。
 * 検出確率モデルとノイズモデルを含みます。
 */
class RadarSimulator {
public:
    /**
     * @brief コンストラクタ
     * @param target_gen 目標生成器
     * @param params レーダーパラメータ
     * @param sensor_id センサーID
     */
    RadarSimulator(const TargetGenerator& target_gen,
                   const RadarParameters& params = RadarParameters(),
                   int sensor_id = 0);

    /**
     * @brief 指定時刻の観測データを生成
     * @param time 時刻 [s]
     * @return 観測データのリスト（目標観測 + クラッタ）
     */
    std::vector<Measurement> generate(double time);

    /**
     * @brief 観測データと真値の対応を取得（評価用）
     * @return 各観測の真の目標ID（-1はクラッタ）
     */
    const std::vector<int>& getTrueAssociations() const {
        return true_associations_;
    }

    /**
     * @brief 真の目標状態を取得（評価用）
     * @param time 時刻 [s]
     * @return 真の状態ベクトル
     */
    std::vector<StateVector> getTrueStates(double time) const;

    /**
     * @brief 検出確率を設定
     */
    void setDetectionProbability(float pd) {
        params_.detection_probability = pd;
    }

    /**
     * @brief クラッタ密度を設定
     */
    void setClutterDensity(float lambda) {
        params_.false_alarm_rate = lambda;
    }

    /**
     * @brief レーダーパラメータを取得
     */
    const RadarParameters& getParams() const { return params_; }

    /**
     * @brief 統計情報をリセット
     */
    void resetStatistics();

    /**
     * @brief 統計情報を表示
     */
    void printStatistics() const;

private:
    const TargetGenerator& target_gen_;
    RadarParameters params_;
    int sensor_id_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<float> uniform_dist_;
    mutable std::normal_distribution<float> normal_dist_;
    mutable std::poisson_distribution<int> poisson_dist_;

    // 評価用
    std::vector<int> true_associations_;  // 各観測の真の目標ID（-1はクラッタ）

    // 統計情報
    int total_targets_;
    int total_detections_;
    int total_clutter_;
    int total_missed_;

    /**
     * @brief 状態ベクトルからレーダー観測を計算
     * @param state 目標状態
     * @return 理想的なレーダー観測（ノイズなし）
     */
    Measurement stateToMeasurement(const StateVector& state) const;

    /**
     * @brief 観測にノイズを付加
     * @param meas 観測データ
     */
    void addNoise(Measurement& meas);

    /**
     * @brief クラッタを生成
     * @param time 時刻 [s]
     * @return クラッタ観測のリスト
     */
    std::vector<Measurement> generateClutter(double time);

    /**
     * @brief 目標が検出されるかどうかを判定
     * @param state 目標状態
     * @return 検出される場合true
     */
    bool isDetected(const StateVector& state) const;

    /**
     * @brief レーダー視野内にあるかチェック
     * @param state 目標状態
     * @return 視野内の場合true
     */
    bool isInFieldOfView(const StateVector& state) const;
};

} // namespace fasttracker

#endif // FASTTRACKER_RADAR_SIMULATOR_HPP
