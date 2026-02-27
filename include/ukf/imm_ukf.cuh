#ifndef FASTTRACKER_IMM_UKF_CUH
#define FASTTRACKER_IMM_UKF_CUH

#include <cuda_runtime.h>
#include "utils/types.hpp"
#include "ukf/ukf.cuh"

namespace fasttracker {

/**
 * @brief IMM（Interacting Multiple Model）GPUフィルタ
 *
 * 複数の運動モデルを並行使用し、モデル確率を動的に更新
 * GPU並列化により高速化
 */
class IMMFilterGPU {
public:
    /**
     * @brief コンストラクタ
     * @param num_models モデル数（通常3: 等速度、高加速度、中加速度）
     * @param max_targets 最大目標数
     */
    IMMFilterGPU(int num_models, int max_targets);
    ~IMMFilterGPU();

    /**
     * @brief 予測ステップ（GPU並列化）
     */
    void predict(const std::vector<StateVector>& states,
                const std::vector<StateCov>& covariances,
                const std::vector<float>& model_probs,
                std::vector<StateVector>& predicted_states,
                std::vector<StateCov>& predicted_covs,
                std::vector<float>& updated_probs,
                int num_targets,
                float dt);

    int getNumModels() const { return num_models_; }

private:
    int num_models_;
    int max_targets_;

    // 各モデル用のUKF（GPU版）
    std::vector<std::unique_ptr<UKF>> model_ukfs_;

    // モデル遷移確率行列（デバイスメモリ）
    float* d_transition_matrix_;
    std::vector<std::vector<float>> h_transition_matrix_;

    // プロセスノイズ（各モデル用）
    std::vector<ProcessNoise> process_noises_;

    // デバイスメモリ（常駐）
    float* d_model_probs_;          // [num_targets * num_models]
    float* d_mixing_probs_;         // [num_targets * num_models * num_models]
    StateVector* d_mixed_states_;   // [num_targets * num_models]
    StateCov* d_mixed_covs_;        // [num_targets * num_models]
    StateVector* d_pred_states_;    // [num_targets * num_models]
    StateCov* d_pred_covs_;         // [num_targets * num_models]
    float* d_likelihoods_;          // [num_targets * num_models]

    // 入出力用デバイスメモリ
    StateVector* d_input_states_;   // [num_targets]
    StateCov* d_input_covs_;        // [num_targets]
    StateVector* d_output_states_;  // [num_targets]
    StateCov* d_output_covs_;       // [num_targets]

    // CUDAストリーム（モデル並列化用）
    cudaStream_t streams_[4];

    /**
     * @brief デバイスメモリ確保
     */
    void allocateDeviceMemory();

    /**
     * @brief デバイスメモリ解放
     */
    void freeDeviceMemory();
};

/**
 * @brief モデル混合確率を計算（GPUカーネル）
 * @param d_model_probs 入力モデル確率 [num_targets * num_models]
 * @param d_transition_matrix 遷移確率行列 [num_models * num_models]
 * @param d_mixing_probs 出力混合確率 [num_targets * num_models * num_models]
 * @param num_targets 目標数
 * @param num_models モデル数
 */
__global__ void computeMixingProbabilitiesKernel(
    const float* d_model_probs,
    const float* d_transition_matrix,
    float* d_mixing_probs,
    int num_targets,
    int num_models
);

/**
 * @brief モデル混合（状態と共分散）（GPUカーネル）
 * @param d_states 入力状態 [num_targets * STATE_DIM]
 * @param d_covs 入力共分散 [num_targets * STATE_DIM * STATE_DIM]
 * @param d_mixing_probs 混合確率 [num_targets * num_models * num_models]
 * @param d_mixed_states 出力混合状態 [num_targets * num_models * STATE_DIM]
 * @param d_mixed_covs 出力混合共分散 [num_targets * num_models * STATE_DIM * STATE_DIM]
 * @param num_targets 目標数
 * @param num_models モデル数
 */
__global__ void mixModelsKernel(
    const StateVector* d_states,
    const StateCov* d_covs,
    const float* d_mixing_probs,
    StateVector* d_mixed_states,
    StateCov* d_mixed_covs,
    int num_targets,
    int num_models
);

/**
 * @brief モデル確率を更新（GPUカーネル）
 * @param d_prior_probs 事前確率 [num_targets * num_models]
 * @param d_likelihoods 尤度 [num_targets * num_models]
 * @param d_posterior_probs 事後確率 [num_targets * num_models]
 * @param num_targets 目標数
 * @param num_models モデル数
 */
__global__ void updateModelProbabilitiesKernel(
    const float* d_prior_probs,
    const float* d_likelihoods,
    float* d_posterior_probs,
    int num_targets,
    int num_models
);

/**
 * @brief 推定値を結合（GPUカーネル）
 * @param d_model_states モデル別状態 [num_targets * num_models * STATE_DIM]
 * @param d_model_covs モデル別共分散 [num_targets * num_models * STATE_DIM * STATE_DIM]
 * @param d_model_probs モデル確率 [num_targets * num_models]
 * @param d_combined_states 結合状態 [num_targets * STATE_DIM]
 * @param d_combined_covs 結合共分散 [num_targets * STATE_DIM * STATE_DIM]
 * @param num_targets 目標数
 * @param num_models モデル数
 */
__global__ void combineEstimatesKernel(
    const StateVector* d_model_states,
    const StateCov* d_model_covs,
    const float* d_model_probs,
    StateVector* d_combined_states,
    StateCov* d_combined_covs,
    int num_targets,
    int num_models
);

} // namespace fasttracker

#endif // FASTTRACKER_IMM_UKF_CUH
