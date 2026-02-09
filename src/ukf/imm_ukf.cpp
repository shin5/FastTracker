#include "ukf/imm_ukf.hpp"
#include <cmath>
#include <algorithm>

namespace fasttracker {

IMMFilter::IMMFilter(int num_models, int max_targets)
    : num_models_(num_models), max_targets_(max_targets) {

    // 各モデル用のUKF初期化
    UKFParams ukf_params;
    MeasurementNoise meas_noise;

    // モデル1: 等速度モデル（巡航・中間飛翔）
    ProcessNoise noise1;
    noise1.position_noise = 2.0f;
    noise1.velocity_noise = 1.0f;
    noise1.accel_noise = 0.5f;

    // モデル2: 高加速度モデル（ブースト段階）
    ProcessNoise noise2;
    noise2.position_noise = 15.0f;
    noise2.velocity_noise = 30.0f;
    noise2.accel_noise = 20.0f;  // 高加速度対応

    // モデル3: 中程度加速度モデル（終末・機動）
    ProcessNoise noise3;
    noise3.position_noise = 8.0f;
    noise3.velocity_noise = 15.0f;
    noise3.accel_noise = 8.0f;

    process_noises_.push_back(noise1);
    process_noises_.push_back(noise2);
    process_noises_.push_back(noise3);

    // 各モデルのUKF生成
    for (int i = 0; i < num_models_; i++) {
        model_ukfs_.push_back(
            std::make_unique<UKF>(max_targets, ukf_params,
                                 process_noises_[i], meas_noise)
        );
    }

    // モデル遷移確率行列（3x3）
    // 各行は [等速度へ, 高加速度へ, 中加速度へ] の遷移確率
    transition_matrix_ = {
        {0.85f, 0.10f, 0.05f},  // 等速度から
        {0.05f, 0.90f, 0.05f},  // 高加速度から
        {0.10f, 0.05f, 0.85f}   // 中加速度から
    };
}

void IMMFilter::predict(
    const std::vector<StateVector>& states,
    const std::vector<StateCov>& covariances,
    const std::vector<float>& model_probs,
    std::vector<StateVector>& predicted_states,
    std::vector<StateCov>& predicted_covs,
    std::vector<float>& updated_probs,
    int num_targets,
    float dt)
{
    // 簡易版: モデル確率に応じて単一の予測を生成
    // （完全版では各モデルで予測し、混合する）

    predicted_states.resize(num_targets);
    predicted_covs.resize(num_targets);
    updated_probs.resize(num_targets * num_models_);

    // 各目標に対してモデル確率が最大のモデルを使用
    for (int i = 0; i < num_targets; i++) {
        // この目標のモデル確率を取得
        int max_model = 0;
        float max_prob = 0.0f;

        for (int m = 0; m < num_models_; m++) {
            int idx = i * num_models_ + m;
            if (idx < static_cast<int>(model_probs.size())) {
                if (model_probs[idx] > max_prob) {
                    max_prob = model_probs[idx];
                    max_model = m;
                }
            }
        }

        // 最大確率のモデルで予測
        // 簡易実装: 等加速度運動モデル
        StateVector pred = states[i];
        pred(0) += pred(2) * dt + 0.5f * pred(4) * dt * dt;  // x
        pred(1) += pred(3) * dt + 0.5f * pred(5) * dt * dt;  // y
        pred(2) += pred(4) * dt;  // vx
        pred(3) += pred(5) * dt;  // vy
        // ax, ay は維持

        predicted_states[i] = pred;
        predicted_covs[i] = covariances[i];

        // プロセスノイズを共分散に追加
        const auto& noise = process_noises_[max_model];
        predicted_covs[i](0, 0) += noise.position_noise * noise.position_noise * dt * dt;
        predicted_covs[i](1, 1) += noise.position_noise * noise.position_noise * dt * dt;
        predicted_covs[i](2, 2) += noise.velocity_noise * noise.velocity_noise * dt * dt;
        predicted_covs[i](3, 3) += noise.velocity_noise * noise.velocity_noise * dt * dt;
        predicted_covs[i](4, 4) += noise.accel_noise * noise.accel_noise * dt * dt;
        predicted_covs[i](5, 5) += noise.accel_noise * noise.accel_noise * dt * dt;

        // モデル確率の更新（簡易版: 遷移確率を適用）
        for (int m = 0; m < num_models_; m++) {
            int idx = i * num_models_ + m;
            if (idx < static_cast<int>(updated_probs.size())) {
                // 遷移確率を使って更新
                float new_prob = 0.0f;
                for (int k = 0; k < num_models_; k++) {
                    int old_idx = i * num_models_ + k;
                    if (old_idx < static_cast<int>(model_probs.size())) {
                        new_prob += model_probs[old_idx] * transition_matrix_[k][m];
                    }
                }
                updated_probs[idx] = new_prob;
            }
        }
    }
}

void IMMFilter::update(
    const std::vector<StateVector>& predicted_states,
    const std::vector<StateCov>& predicted_covs,
    const std::vector<Measurement>& measurements,
    const std::vector<float>& model_probs,
    std::vector<StateVector>& updated_states,
    std::vector<StateCov>& updated_covs,
    std::vector<float>& updated_probs)
{
    // 簡易版: 予測値をそのまま使用
    // （完全版では測定値で更新し、尤度を計算してモデル確率を更新）
    updated_states = predicted_states;
    updated_covs = predicted_covs;
    updated_probs = model_probs;
}

void IMMFilter::mixModels(
    const std::vector<StateVector>& states,
    const std::vector<StateCov>& covariances,
    const std::vector<float>& model_probs,
    std::vector<StateVector>& mixed_states,
    std::vector<StateCov>& mixed_covs)
{
    // 簡易版では実装省略
}

void IMMFilter::updateModelProbabilities(
    const std::vector<float>& likelihoods,
    const std::vector<float>& prior_probs,
    std::vector<float>& posterior_probs)
{
    // 簡易版では実装省略
}

void IMMFilter::combineEstimates(
    const std::vector<std::vector<StateVector>>& model_states,
    const std::vector<std::vector<StateCov>>& model_covs,
    const std::vector<float>& model_probs,
    std::vector<StateVector>& combined_states,
    std::vector<StateCov>& combined_covs)
{
    // 簡易版では実装省略
}

} // namespace fasttracker
