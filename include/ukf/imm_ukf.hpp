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
 * 複数の運動モデルを並行使用し、モデル確率を動的に更新
 * 弾道ミサイルの多段階飛翔に対応
 */
class IMMFilter {
public:
    /**
     * @brief コンストラクタ
     * @param num_models モデル数（通常3: 等速度、高加速度、重力）
     * @param max_targets 最大目標数
     */
    IMMFilter(int num_models, int max_targets);

    /**
     * @brief 予測ステップ（全モデル）
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

    int getNumModels() const { return num_models_; }

private:
    int num_models_;
    int max_targets_;

    // 各モデル用のUKF
    std::vector<std::unique_ptr<UKF>> model_ukfs_;

    // モデル遷移確率行列
    std::vector<std::vector<float>> transition_matrix_;

    // プロセスノイズ（各モデル用）
    std::vector<ProcessNoise> process_noises_;

    /**
     * @brief モデル間の混合
     */
    void mixModels(const std::vector<StateVector>& states,
                  const std::vector<StateCov>& covariances,
                  const std::vector<float>& model_probs,
                  std::vector<StateVector>& mixed_states,
                  std::vector<StateCov>& mixed_covs);

    /**
     * @brief モデル確率の更新
     */
    void updateModelProbabilities(const std::vector<float>& likelihoods,
                                 const std::vector<float>& prior_probs,
                                 std::vector<float>& posterior_probs);

    /**
     * @brief 推定値の結合
     */
    void combineEstimates(const std::vector<std::vector<StateVector>>& model_states,
                         const std::vector<std::vector<StateCov>>& model_covs,
                         const std::vector<float>& model_probs,
                         std::vector<StateVector>& combined_states,
                         std::vector<StateCov>& combined_covs);
};

} // namespace fasttracker

#endif // FASTTRACKER_IMM_UKF_HPP
