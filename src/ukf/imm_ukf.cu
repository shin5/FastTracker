#include "ukf/imm_ukf.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>
#include <iostream>

namespace fasttracker {

// ========================================
// CUDAカーネル実装
// ========================================

__global__ void computeMixingProbabilitiesKernel(
    const float* d_model_probs,
    const float* d_transition_matrix,
    float* d_mixing_probs,
    int num_targets,
    int num_models)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int target_idx = tid / (num_models * num_models);

    if (target_idx >= num_targets) return;

    int i = (tid / num_models) % num_models;  // 遷移先モデル
    int j = tid % num_models;                  // 遷移元モデル

    // 混合確率 = P(M_j) * P(M_i|M_j) / c_i
    // c_i = Σ_j P(M_j) * P(M_i|M_j)

    float prior_j = d_model_probs[target_idx * num_models + j];
    float trans_ji = d_transition_matrix[j * num_models + i];

    // 正規化定数を計算（各iに対して）
    float c_i = 0.0f;
    for (int k = 0; k < num_models; k++) {
        float prior_k = d_model_probs[target_idx * num_models + k];
        float trans_ki = d_transition_matrix[k * num_models + i];
        c_i += prior_k * trans_ki;
    }

    float mixing_prob = (c_i > 1e-6f) ? (prior_j * trans_ji / c_i) : (1.0f / num_models);
    d_mixing_probs[target_idx * num_models * num_models + i * num_models + j] = mixing_prob;
}

__global__ void mixModelsKernel(
    const StateVector* d_states,
    const StateCov* d_covs,
    const float* d_mixing_probs,
    StateVector* d_mixed_states,
    StateCov* d_mixed_covs,
    int num_targets,
    int num_models)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int target_idx = tid / num_models;
    int model_i = tid % num_models;

    if (target_idx >= num_targets) return;

    // 各モデルiに対する混合状態を計算
    StateVector mixed_state;
    mixed_state.setZero();

    for (int j = 0; j < num_models; j++) {
        float mixing_prob = d_mixing_probs[target_idx * num_models * num_models + model_i * num_models + j];
        const StateVector& state_j = d_states[target_idx];

        for (int k = 0; k < STATE_DIM; k++) {
            mixed_state(k) += mixing_prob * state_j(k);
        }
    }

    d_mixed_states[target_idx * num_models + model_i] = mixed_state;

    // 混合共分散を計算
    StateCov mixed_cov;
    mixed_cov.setZero();

    for (int j = 0; j < num_models; j++) {
        float mixing_prob = d_mixing_probs[target_idx * num_models * num_models + model_i * num_models + j];
        const StateVector& state_j = d_states[target_idx];
        const StateCov& cov_j = d_covs[target_idx];

        // P_ij = P_j + (x_j - x_0i)(x_j - x_0i)^T
        StateVector diff = state_j - mixed_state;

        for (int row = 0; row < STATE_DIM; row++) {
            for (int col = 0; col < STATE_DIM; col++) {
                mixed_cov(row, col) += mixing_prob * (cov_j(row, col) + diff(row) * diff(col));
            }
        }
    }

    d_mixed_covs[target_idx * num_models + model_i] = mixed_cov;
}

__global__ void updateModelProbabilitiesKernel(
    const float* d_prior_probs,
    const float* d_likelihoods,
    float* d_posterior_probs,
    int num_targets,
    int num_models)
{
    int target_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_idx >= num_targets) return;

    // Bayes更新: P(M_i|z) = L_i * P(M_i) / Σ_j L_j * P(M_j)
    float normalization = 0.0f;

    for (int i = 0; i < num_models; i++) {
        float prior = d_prior_probs[target_idx * num_models + i];
        float likelihood = d_likelihoods[target_idx * num_models + i];
        normalization += likelihood * prior;
    }

    for (int i = 0; i < num_models; i++) {
        float prior = d_prior_probs[target_idx * num_models + i];
        float likelihood = d_likelihoods[target_idx * num_models + i];

        float posterior = (normalization > 1e-10f) ?
            (likelihood * prior / normalization) : (1.0f / num_models);

        d_posterior_probs[target_idx * num_models + i] = posterior;
    }
}

__global__ void combineEstimatesKernel(
    const StateVector* d_model_states,
    const StateCov* d_model_covs,
    const float* d_model_probs,
    StateVector* d_combined_states,
    StateCov* d_combined_covs,
    int num_targets,
    int num_models)
{
    int target_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_idx >= num_targets) return;

    // 加重平均で状態を結合
    StateVector combined_state;
    combined_state.setZero();

    for (int i = 0; i < num_models; i++) {
        float prob = d_model_probs[target_idx * num_models + i];
        const StateVector& state_i = d_model_states[target_idx * num_models + i];

        for (int k = 0; k < STATE_DIM; k++) {
            combined_state(k) += prob * state_i(k);
        }
    }

    d_combined_states[target_idx] = combined_state;

    // 加重平均で共分散を結合（スプレッド項含む）
    StateCov combined_cov;
    combined_cov.setZero();

    for (int i = 0; i < num_models; i++) {
        float prob = d_model_probs[target_idx * num_models + i];
        const StateVector& state_i = d_model_states[target_idx * num_models + i];
        const StateCov& cov_i = d_model_covs[target_idx * num_models + i];

        StateVector diff = state_i - combined_state;

        for (int row = 0; row < STATE_DIM; row++) {
            for (int col = 0; col < STATE_DIM; col++) {
                combined_cov(row, col) += prob * (cov_i(row, col) + diff(row) * diff(col));
            }
        }
    }

    d_combined_covs[target_idx] = combined_cov;
}

// ========================================
// IMMFilterGPUクラス実装
// ========================================

IMMFilterGPU::IMMFilterGPU(int num_models, int max_targets)
    : num_models_(num_models), max_targets_(max_targets),
      d_transition_matrix_(nullptr),
      d_model_probs_(nullptr),
      d_mixing_probs_(nullptr),
      d_mixed_states_(nullptr),
      d_mixed_covs_(nullptr),
      d_pred_states_(nullptr),
      d_pred_covs_(nullptr),
      d_likelihoods_(nullptr),
      d_input_states_(nullptr),
      d_input_covs_(nullptr),
      d_output_states_(nullptr),
      d_output_covs_(nullptr)
{
    // 各モデル用のUKF初期化
    UKFParams ukf_params;
    MeasurementNoise meas_noise;

    // デフォルトの外部ノイズ基準値
    ProcessNoise ext_noise;  // デフォルト値を使用

    // Model 0 (CV): 安定飛翔 — 外部ノイズの10%
    ProcessNoise noise1;
    noise1.position_noise = ext_noise.position_noise * 0.1f;
    noise1.velocity_noise = ext_noise.velocity_noise * 0.1f;
    noise1.accel_noise = ext_noise.accel_noise * 0.1f;

    // Model 1 (Ballistic): 物理モデルが正確なので低め — 外部ノイズの30%
    ProcessNoise noise2;
    noise2.position_noise = ext_noise.position_noise * 0.3f;
    noise2.velocity_noise = ext_noise.velocity_noise * 0.3f;
    noise2.accel_noise = ext_noise.accel_noise * 0.3f;

    // Model 2 (CT): 機動中 — 外部ノイズの100%
    ProcessNoise noise3;
    noise3.position_noise = ext_noise.position_noise * 1.0f;
    noise3.velocity_noise = ext_noise.velocity_noise * 1.0f;
    noise3.accel_noise = ext_noise.accel_noise * 1.0f;

    process_noises_.push_back(noise1);
    process_noises_.push_back(noise2);
    process_noises_.push_back(noise3);

    // 各モデルのUKF生成（GPU版）
    for (int i = 0; i < num_models_; i++) {
        model_ukfs_.push_back(
            std::make_unique<UKF>(max_targets, ukf_params,
                                 process_noises_[i], meas_noise)
        );
    }

    // モデル遷移確率行列（CPU版と同一）
    //        → CV    → Bal   → CT
    // CV    [0.80   0.15    0.05]
    // Bal   [0.10   0.85    0.05]
    // CT    [0.05   0.10    0.85]
    h_transition_matrix_ = {
        {0.80f, 0.15f, 0.05f},  // CVから
        {0.10f, 0.85f, 0.05f},  // Ballisticから
        {0.05f, 0.10f, 0.85f}   // CTから
    };

    // CUDAストリーム作成（モデル並列化用）
    for (int i = 0; i < num_models_; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }

    // デバイスメモリ確保
    allocateDeviceMemory();

    // 遷移行列をデバイスにコピー
    std::vector<float> flat_matrix;
    for (const auto& row : h_transition_matrix_) {
        for (float val : row) {
            flat_matrix.push_back(val);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_transition_matrix_, flat_matrix.data(),
                          num_models_ * num_models_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::cout << "IMMFilterGPU initialized with " << num_models_ << " models (GPU-accelerated)" << std::endl;
    std::cout << "  Using " << num_models_ << " CUDA streams for parallel execution" << std::endl;
}

IMMFilterGPU::~IMMFilterGPU() {
    freeDeviceMemory();

    // CUDAストリーム破棄
    for (int i = 0; i < num_models_; i++) {
        cudaStreamDestroy(streams_[i]);
    }
}

void IMMFilterGPU::allocateDeviceMemory() {
    CUDA_CHECK(cudaMalloc(&d_transition_matrix_, num_models_ * num_models_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_model_probs_, max_targets_ * num_models_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mixing_probs_, max_targets_ * num_models_ * num_models_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mixed_states_, max_targets_ * num_models_ * sizeof(StateVector)));
    CUDA_CHECK(cudaMalloc(&d_mixed_covs_, max_targets_ * num_models_ * sizeof(StateCov)));
    CUDA_CHECK(cudaMalloc(&d_pred_states_, max_targets_ * num_models_ * sizeof(StateVector)));
    CUDA_CHECK(cudaMalloc(&d_pred_covs_, max_targets_ * num_models_ * sizeof(StateCov)));
    CUDA_CHECK(cudaMalloc(&d_likelihoods_, max_targets_ * num_models_ * sizeof(float)));

    // 入出力用バッファ
    CUDA_CHECK(cudaMalloc(&d_input_states_, max_targets_ * sizeof(StateVector)));
    CUDA_CHECK(cudaMalloc(&d_input_covs_, max_targets_ * sizeof(StateCov)));
    CUDA_CHECK(cudaMalloc(&d_output_states_, max_targets_ * sizeof(StateVector)));
    CUDA_CHECK(cudaMalloc(&d_output_covs_, max_targets_ * sizeof(StateCov)));
}

void IMMFilterGPU::freeDeviceMemory() {
    if (d_transition_matrix_) cudaFree(d_transition_matrix_);
    if (d_model_probs_) cudaFree(d_model_probs_);
    if (d_mixing_probs_) cudaFree(d_mixing_probs_);
    if (d_mixed_states_) cudaFree(d_mixed_states_);
    if (d_mixed_covs_) cudaFree(d_mixed_covs_);
    if (d_pred_states_) cudaFree(d_pred_states_);
    if (d_pred_covs_) cudaFree(d_pred_covs_);
    if (d_likelihoods_) cudaFree(d_likelihoods_);
    if (d_input_states_) cudaFree(d_input_states_);
    if (d_input_covs_) cudaFree(d_input_covs_);
    if (d_output_states_) cudaFree(d_output_states_);
    if (d_output_covs_) cudaFree(d_output_covs_);
}

void IMMFilterGPU::predict(
    const std::vector<StateVector>& states,
    const std::vector<StateCov>& covariances,
    const std::vector<float>& model_probs,
    std::vector<StateVector>& predicted_states,
    std::vector<StateCov>& predicted_covs,
    std::vector<float>& updated_probs,
    int num_targets,
    float dt)
{
    if (num_targets == 0) return;

    // ========================================
    // 最適化: 単一のメモリ転送（H2D）
    // ========================================
    CUDA_CHECK(cudaMemcpy(d_input_states_, states.data(),
                          num_targets * sizeof(StateVector),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_covs_, covariances.data(),
                          num_targets * sizeof(StateCov),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_model_probs_, model_probs.data(),
                          num_targets * num_models_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int num_blocks = (num_targets + threads_per_block - 1) / threads_per_block;

    // ========================================
    // ステップ1: 各モデルで並列予測（CUDAストリーム使用）
    // ========================================
    for (int m = 0; m < num_models_; m++) {
        // 各モデルのUKFにデータをコピー（非同期、ストリーム利用）
        CUDA_CHECK(cudaMemcpyAsync(model_ukfs_[m]->getDeviceStates(), d_input_states_,
                                    num_targets * sizeof(StateVector),
                                    cudaMemcpyDeviceToDevice, streams_[m]));
        CUDA_CHECK(cudaMemcpyAsync(model_ukfs_[m]->getDeviceCovariances(), d_input_covs_,
                                    num_targets * sizeof(StateCov),
                                    cudaMemcpyDeviceToDevice, streams_[m]));

        // 予測実行（各モデルのmodel_idを渡す: 0=CV, 1=Ballistic, 2=CT）
        model_ukfs_[m]->predict(model_ukfs_[m]->getDeviceStates(),
                               model_ukfs_[m]->getDeviceCovariances(),
                               num_targets, dt, m);

        // 結果をd_pred_states_, d_pred_covs_にコピー（非同期）
        for (int t = 0; t < num_targets; t++) {
            CUDA_CHECK(cudaMemcpyAsync(&d_pred_states_[t * num_models_ + m],
                                        &(model_ukfs_[m]->getDeviceStates())[t],
                                        sizeof(StateVector),
                                        cudaMemcpyDeviceToDevice, streams_[m]));
            CUDA_CHECK(cudaMemcpyAsync(&d_pred_covs_[t * num_models_ + m],
                                        &(model_ukfs_[m]->getDeviceCovariances())[t],
                                        sizeof(StateCov),
                                        cudaMemcpyDeviceToDevice, streams_[m]));
        }
    }

    // 全ストリーム同期
    for (int m = 0; m < num_models_; m++) {
        CUDA_CHECK(cudaStreamSynchronize(streams_[m]));
    }

    // ========================================
    // ステップ2: 尤度初期化（簡易版: 均等）
    // ========================================
    CUDA_CHECK(cudaMemset(d_likelihoods_, 0, num_targets * num_models_ * sizeof(float)));
    std::vector<float> h_likelihoods(num_targets * num_models_, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_likelihoods_, h_likelihoods.data(),
                          num_targets * num_models_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ========================================
    // ステップ3: モデル確率を更新
    // ========================================
    updateModelProbabilitiesKernel<<<num_blocks, threads_per_block>>>(
        d_model_probs_, d_likelihoods_, d_model_probs_,
        num_targets, num_models_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================
    // ステップ4: 推定値を結合（デバイス上で直接）
    // ========================================
    combineEstimatesKernel<<<num_blocks, threads_per_block>>>(
        d_pred_states_, d_pred_covs_, d_model_probs_,
        d_output_states_, d_output_covs_,
        num_targets, num_models_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================
    // 最適化: 単一のメモリ転送（D2H）
    // ========================================
    predicted_states.resize(num_targets);
    predicted_covs.resize(num_targets);
    updated_probs.resize(num_targets * num_models_);

    CUDA_CHECK(cudaMemcpy(predicted_states.data(), d_output_states_,
                          num_targets * sizeof(StateVector),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(predicted_covs.data(), d_output_covs_,
                          num_targets * sizeof(StateCov),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(updated_probs.data(), d_model_probs_,
                          num_targets * num_models_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

} // namespace fasttracker
