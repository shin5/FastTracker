#ifndef FASTTRACKER_RADAR_SIMULATOR_HPP
#define FASTTRACKER_RADAR_SIMULATOR_HPP

#include <vector>
#include <random>
#include <iostream>
#include "utils/types.hpp"
#include "target_generator.hpp"

namespace fasttracker {

/**
 * @brief レーダーパラメータ
 */
struct RadarParameters {
    float detection_probability;    // 検出確率 Pd
    float false_alarm_rate;         // 誤警報率（クラッタ密度） [1/m²]
    float min_range;                // 最小探知距離 [m]（この距離以下は検出しない）
    float max_range;                // 最大探知距離 [m]
    float azimuth_coverage;         // 方位覆域幅 [rad] - Total azimuth coverage width
    float min_elevation;            // 最小仰角 [rad] - Physical lower elevation limit
    float max_elevation;            // 最大仰角 [rad] - Physical upper elevation limit
    float field_of_view;            // [DEPRECATED] 視野角 [rad] - Use azimuth_coverage instead
    float snr_ref;                  // 基準SN比 [dB] @ 1km
    float sensor_x;                 // センサーX座標 [m]
    float sensor_y;                 // センサーY座標 [m]
    float sensor_z;                 // センサーZ座標（高度） [m]
    MeasurementNoise meas_noise;    // 観測ノイズ

    // ビームステアリング
    float beam_width;                // ビーム角度幅 [rad] (~3°)
    int num_beams;                   // フレームあたりのビーム数
    float antenna_boresight;         // アンテナ中心方位 [rad] (atan2系: 0=East, π/2=North)
    float pfa_per_pulse;             // パルスあたり誤警報確率 (CFAR閾値算出用, Swerling II)
    float pd_at_ref_range;           // 基準距離における検出確率（SNR Ref自動計算基準）
    float det_ref_range_m;           // 検出性能基準距離 [m]（例: 10000 = 10 km）
    float search_min_range;          // サーチ領域最小距離 [m]
    float search_max_range;          // サーチ領域最大距離 [m] (0 = max_range と同じ)
    float track_range_width;         // 追尾ビーム距離幅 [m] (目標距離±width/2, 0 = 制限なし)
    float range_resolution;          // レンジ分解能 [m] (同一ビーム内でこの距離以内の目標は未分解)

    RadarParameters()
        : detection_probability(0.95f),
          false_alarm_rate(1e-8f),
          min_range(0.0f),
          max_range(20000.0f),
          azimuth_coverage(2.0f * static_cast<float>(M_PI)),  // 360° omnidirectional
          min_elevation(-0.5236f),                             // -30°
          max_elevation(1.5708f),                              // +90° (zenith)
          field_of_view(2.0f * static_cast<float>(M_PI)),     // DEPRECATED: same as azimuth_coverage
          snr_ref(60.0f),
          sensor_x(0.0f),
          sensor_y(0.0f),
          sensor_z(0.0f),
          beam_width(0.052f),
          num_beams(10),
          antenna_boresight(0.0f),
          pfa_per_pulse(1e-6f),
          pd_at_ref_range(0.9f),
          det_ref_range_m(10000.0f),
          search_min_range(0.0f),
          search_max_range(0.0f),
          track_range_width(0.0f),
          range_resolution(150.0f) {}

    /**
     * @brief Swerling II / P_FA / 基準距離 から SNR Ref を物理整合的に自動計算する
     *
     * 基準距離 det_ref_range_m で P(D) = pd_at_ref_range となる平均SNRを算出し、
     * R^4則で 1km基準に逆算して snr_ref [dB] を設定する。
     *
     *   γ_T = −ln(pfa_per_pulse)
     *   SNR_avg(R_ref) = γ_T / (−ln(pd_at_ref_range))
     *   snr_ref = 10·log10(SNR_avg(R_ref)) + 40·log10(R_ref / 1000)
     */
    void computeSnrRef() {
        float gamma_T    = -std::log(std::max(pfa_per_pulse, 1e-30f));
        float pd_clamped = std::max(std::min(pd_at_ref_range, 0.9999f), 1e-4f);
        float snr_avg_lin = gamma_T / (-std::log(pd_clamped));
        float snr_avg_dB  = 10.0f * std::log10(std::max(snr_avg_lin, 1e-10f));
        float range_correction = 40.0f * std::log10(det_ref_range_m / 1000.0f);
        snr_ref = snr_avg_dB + range_correction;

        // 診断出力
        std::cout << "\n[SNR Ref Calculation]" << std::endl;
        std::cout << "  pfa_per_pulse: " << pfa_per_pulse << std::endl;
        std::cout << "  pd_at_ref_range: " << pd_at_ref_range << std::endl;
        std::cout << "  det_ref_range_m: " << det_ref_range_m << " m (" << (det_ref_range_m/1000.0f) << " km)" << std::endl;
        std::cout << "  gamma_T (CFAR threshold): " << gamma_T << std::endl;
        std::cout << "  Required SNR (linear): " << snr_avg_lin << std::endl;
        std::cout << "  Required SNR at " << (det_ref_range_m/1000.0f) << " km: " << snr_avg_dB << " dB" << std::endl;
        std::cout << "  Range correction (40*log10(" << (det_ref_range_m/1000.0f) << "/1)): " << range_correction << " dB" << std::endl;
        std::cout << "  SNR Ref @ 1km: " << snr_ref << " dB" << std::endl;
    }
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
     * @brief 各観測のビーム種別を取得
     * @return 0=サーチビーム, 1=追尾ビーム
     */
    const std::vector<int>& getBeamTypes() const {
        return beam_types_;
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
     * @brief 乱数シードを設定
     */
    void setSeed(uint32_t seed) { rng_.seed(seed); }

    /**
     * @brief ビーム方向を設定（ビームステアリングモード用）
     * @param directions ビーム中心方位角のリスト [rad]
     */
    void setBeamDirections(const std::vector<float>& directions) {
        beam_directions_ = directions;
    }

    /**
     * @brief 現在のビーム方向を取得
     */
    const std::vector<float>& getBeamDirections() const {
        return beam_directions_;
    }

    /**
     * @brief ビーム仰角を設定（beam_directions_ と同じインデックスで対応）
     * @param elevations ビーム中心仰角のリスト [rad]
     */
    void setBeamElevations(const std::vector<float>& elevations) {
        beam_elevations_ = elevations;
    }

    /**
     * @brief 現在のビーム仰角を取得
     */
    const std::vector<float>& getBeamElevations() const {
        return beam_elevations_;
    }

    /**
     * @brief 追尾ビームの目標距離を設定（beam_directions_ と同じインデックス）
     * @param ranges 目標距離のリスト [m]（0=サーチビーム）
     */
    void setBeamTargetRanges(const std::vector<float>& ranges) {
        beam_target_ranges_ = ranges;
    }

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
    std::vector<int> beam_types_;         // 各観測のビーム種別（0=サーチ, 1=追尾）
    std::vector<int> clutter_beam_types_; // generateClutter用の一時バッファ

    // 統計情報
    int total_targets_;
    int total_detections_;
    int total_clutter_;
    int total_missed_;

    // ビームステアリング状態
    std::vector<float> beam_directions_;  // 現フレームのビーム中心方位角 [rad]
    std::vector<float> beam_elevations_;  // 現フレームのビーム中心仰角 [rad]（beam_directions_ と同インデックス）
    std::vector<float> beam_target_ranges_;  // 追尾ビームの目標距離 [m]（0=サーチビーム）

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
     * @brief 目標が検出されるかどうかを判定（レガシー: detection_probability使用）
     * @param state 目標状態
     * @return 検出される場合true
     */
    bool isDetected(const StateVector& state) const;

    /**
     * @brief Swerling II モデルによる検出判定
     * @param snr_inst_dB 瞬時SNR [dB] (Swerling IIサンプリング済み)
     * @return 検出される場合true (CFAR閾値 γ = −ln(pfa_per_pulse) と比較)
     */
    bool isDetectedSwerlingII(float snr_inst_dB) const;

    /**
     * @brief レーダー視野内にあるかチェック
     * @param state 目標状態
     * @return 視野内の場合true
     */
    bool isInFieldOfView(const StateVector& state) const;

    /**
     * @brief 目標がレーダー覆域内にあるかチェック（距離 + 仰角のみ、方位角制限なし）
     * @param state 目標の状態ベクトル
     * @return 覆域内の場合true
     */
    bool isInRadarCoverage(const StateVector& state) const;

    /**
     * @brief 目標がアクティブビーム内にあるかチェック（方位角・仰角の両方で判定）
     * @param azimuth 目標の方位角 [rad]
     * @param elevation 目標の仰角 [rad]
     * @return ビーム内の場合true
     */
    bool isOnBeam(float azimuth, float elevation) const;

    /**
     * @brief ビームインデックス付き判定（ビーム種別特定用）
     * @param azimuth 方位角 [rad]
     * @param elevation 仰角 [rad]
     * @param beam_idx [out] マッチしたビームのインデックス (-1=マッチなし)
     * @return ビーム内の場合true
     */
    bool isOnBeamWithIndex(float azimuth, float elevation, int& beam_idx) const;
};

} // namespace fasttracker

#endif // FASTTRACKER_RADAR_SIMULATOR_HPP
