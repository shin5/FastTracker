#ifndef FASTTRACKER_IMM_UKF_HPP
#define FASTTRACKER_IMM_UKF_HPP

#include <vector>
#include <memory>
#include "ukf.cuh"
#include "utils/types.hpp"

namespace fasttracker {

/**
 * @brief IMM（Interacting Multiple Model）フィルタ
 *
 * 4つの運動モデルを並行使用し、モデル確率を動的に更新
 *   Model 0: CA  （Singer加速度 τ=20s — 汎用持続加速：ブースト/グライド）
 *   Model 1: Ballistic（弾道 — 重力+大気抗力 RK4）
 *   Model 2: CT  （旋回 — HGV機動）
 *   Model 3: SkipGlide（弾道+空力揚力 — HGVスキップ/グライド）
 */
class IMMFilter {
public:
    IMMFilter(int num_models, int max_targets, const ProcessNoise& external_noise);

    /**
     * @brief 予測ステップ（全モデル並行予測 + 重み付き統合）
     */
    void predict(const std::vector<StateVector>& states,
                const std::vector<StateCov>& covariances,
                const std::vector<float>& model_probs,
                std::vector<StateVector>& predicted_states,
                std::vector<StateCov>& predicted_covs,
                std::vector<float>& updated_probs,
                int num_targets,
                float dt);

    /**
     * @brief 更新ステップ（測定値で更新）
     */
    void update(const std::vector<StateVector>& predicted_states,
               const std::vector<StateCov>& predicted_covs,
               const std::vector<Measurement>& measurements,
               const std::vector<float>& model_probs,
               std::vector<StateVector>& updated_states,
               std::vector<StateCov>& updated_covs,
               std::vector<float>& updated_probs);

    /**
     * @brief 観測尤度ベースのモデル確率更新
     *
     * UKF測定更新後、各モデルの予測観測と実観測の一致度（尤度）から
     * モデル確率を更新する。正規IMMサイクルの核心部分。
     *
     * @param track_indices 更新対象トラックのインデックス（predict時の順序）
     * @param measurements  対応する実測値
     * @param model_probs   現在のモデル確率 [num_tracks * num_models]
     * @param updated_probs 更新後のモデル確率（出力）
     * @param sensor_x センサーX [m]
     * @param sensor_y センサーY [m]
     * @param sensor_z センサーZ [m]
     */
    void updateModelProbabilities(
        const std::vector<int>& track_indices,
        const std::vector<Measurement>& measurements,
        std::vector<float>& model_probs,
        float sensor_x, float sensor_y, float sensor_z);

    int getNumModels() const { return num_models_; }

    /**
     * @brief モデル遷移確率行列を設定
     * @param matrix 3x3遷移確率行列
     */
    void setTransitionMatrix(const std::vector<std::vector<float>>& matrix) {
        if (matrix.size() >= 4 && matrix[0].size() >= 4) {
            transition_matrix_ = matrix;
        }
    }

    /**
     * @brief モデル別プロセスノイズ倍率を設定
     * @param ca_mult CA (Singer) モデルのノイズ倍率 (デフォルト: 0.1)
     * @param bal_mult Ballistic モデルのノイズ倍率 (デフォルト: 0.3)
     * @param ct_mult CT モデルのノイズ倍率 (デフォルト: 2.5)
     */
    void setModelNoiseMultipliers(float cv_mult, float bal_mult, float ct_mult, float sg_mult);

private:
    int num_models_;
    int max_targets_;

    // 各モデル用のUKF
    std::vector<std::unique_ptr<UKF>> model_ukfs_;

    // モデル遷移確率行列
    std::vector<std::vector<float>> transition_matrix_;

    // プロセスノイズ（各モデル用）
    std::vector<ProcessNoise> process_noises_;

    // 基準プロセスノイズ（倍率計算用）
    ProcessNoise base_process_noise_;

    // 観測ノイズ（尤度計算用）
    MeasurementNoise meas_noise_;

    // predict()で保存されるモデル別予測状態
    // per_model_predictions_[target_idx][model_idx]
    std::vector<std::array<StateVector, 4>> per_model_predictions_;
    std::vector<std::array<StateCov, 4>> per_model_pred_covs_;

    // CPU版運動モデル予測
    static StateVector predictCA_CPU(const StateVector& state, float dt);
    static StateVector predictBallistic_CPU(const StateVector& state, float dt);
    static StateVector predictCT_CPU(const StateVector& state, float dt);
    static StateVector predictSkipGlide_CPU(const StateVector& state, float dt);

    // 状態→観測への変換（CPU版、尤度計算用）
    static MeasVector stateToMeas_CPU(const StateVector& state,
                                       float sensor_x, float sensor_y, float sensor_z);

    // 状態遷移ヤコビアン近似 (簡易線形化)
    static StateCov computeApproxF(float dt);
};

} // namespace fasttracker

#endif // FASTTRACKER_IMM_UKF_HPP
