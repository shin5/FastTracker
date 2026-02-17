#include "ukf/imm_ukf.hpp"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

// 物理定数（弾道モデル用、GPUの__constant__と同値）
static constexpr float CPU_GRAVITY_G0 = 9.80665f;
static constexpr float CPU_EARTH_RADIUS = 6371000.0f;
static constexpr float CPU_ATM_RHO0 = 1.225f;
static constexpr float CPU_ATM_SCALE_HEIGHT = 7400.0f;
static constexpr float CPU_BALLISTIC_BETA = 0.001f;

IMMFilter::IMMFilter(int num_models, int max_targets, const ProcessNoise& external_noise)
    : num_models_(num_models), max_targets_(max_targets), meas_noise_() {

    UKFParams ukf_params;
    MeasurementNoise meas_noise;

    // Model 0 (CV): 安定飛翔 — 外部ノイズの10%
    ProcessNoise noise_cv;
    noise_cv.position_noise = external_noise.position_noise * 0.1f;
    noise_cv.velocity_noise = external_noise.velocity_noise * 0.1f;
    noise_cv.accel_noise = external_noise.accel_noise * 0.1f;

    // Model 1 (Ballistic): 物理モデルが正確なので低め — 外部ノイズの30%
    ProcessNoise noise_bal;
    noise_bal.position_noise = external_noise.position_noise * 0.3f;
    noise_bal.velocity_noise = external_noise.velocity_noise * 0.3f;
    noise_bal.accel_noise = external_noise.accel_noise * 0.3f;

    // Model 2 (CT): 機動中 — 外部ノイズの100%
    ProcessNoise noise_ct;
    noise_ct.position_noise = external_noise.position_noise * 1.0f;
    noise_ct.velocity_noise = external_noise.velocity_noise * 1.0f;
    noise_ct.accel_noise = external_noise.accel_noise * 1.0f;

    process_noises_.push_back(noise_cv);
    process_noises_.push_back(noise_bal);
    process_noises_.push_back(noise_ct);

    // 各モデルのUKF生成
    for (int i = 0; i < num_models_; i++) {
        model_ukfs_.push_back(
            std::make_unique<UKF>(max_targets, ukf_params,
                                 process_noises_[i], meas_noise)
        );
    }

    // モデル遷移確率行列
    //        → CV    → Bal   → CT
    // CV    [0.80   0.15    0.05]
    // Bal   [0.10   0.85    0.05]
    // CT    [0.05   0.10    0.85]
    transition_matrix_ = {
        {0.80f, 0.15f, 0.05f},  // CVから
        {0.10f, 0.85f, 0.05f},  // Ballisticから
        {0.05f, 0.10f, 0.85f}   // CTから
    };
}

// ================================================================
// CPU版運動モデル（GPUの__device__関数と同一ロジック）
// 状態: [x, y, z, vx, vy, vz, ax, ay, az]
// ================================================================

StateVector IMMFilter::predictCV_CPU(const StateVector& state, float dt) {
    StateVector pred;
    // 位置: 等速直線
    pred(0) = state(0) + state(3) * dt;
    pred(1) = state(1) + state(4) * dt;
    pred(2) = state(2) + state(5) * dt;
    // 速度: 一定
    pred(3) = state(3);
    pred(4) = state(4);
    pred(5) = state(5);
    // 加速度: 減衰 (τ=5s)
    float decay = std::exp(-dt / 5.0f);
    pred(6) = state(6) * decay;
    pred(7) = state(7) * decay;
    pred(8) = state(8) * decay;
    return pred;
}

StateVector IMMFilter::predictBallistic_CPU(const StateVector& state, float dt) {
    float x = state(0), y = state(1), z = state(2);
    float vx = state(3), vy = state(4), vz = state(5);

    // RK4 Stage 1
    float alt1 = std::max(z, 0.0f);
    float gr1 = CPU_EARTH_RADIUS / (CPU_EARTH_RADIUS + alt1);
    float g1 = CPU_GRAVITY_G0 * gr1 * gr1;
    float rho1 = CPU_ATM_RHO0 * std::exp(-alt1 / CPU_ATM_SCALE_HEIGHT);
    float spd1 = std::sqrt(vx*vx + vy*vy + vz*vz);
    float df1 = CPU_BALLISTIC_BETA * rho1 * spd1;
    float k1_vx = -df1*vx, k1_vy = -df1*vy, k1_vz = -g1 - df1*vz;
    float k1_x = vx, k1_y = vy, k1_z = vz;

    // RK4 Stage 2
    float hdt = 0.5f * dt;
    float vx2 = vx + hdt*k1_vx, vy2 = vy + hdt*k1_vy, vz2 = vz + hdt*k1_vz;
    float z2 = z + hdt*k1_z;
    float alt2 = std::max(z2, 0.0f);
    float gr2 = CPU_EARTH_RADIUS / (CPU_EARTH_RADIUS + alt2);
    float g2 = CPU_GRAVITY_G0 * gr2 * gr2;
    float rho2 = CPU_ATM_RHO0 * std::exp(-alt2 / CPU_ATM_SCALE_HEIGHT);
    float spd2 = std::sqrt(vx2*vx2 + vy2*vy2 + vz2*vz2);
    float df2 = CPU_BALLISTIC_BETA * rho2 * spd2;
    float k2_vx = -df2*vx2, k2_vy = -df2*vy2, k2_vz = -g2 - df2*vz2;
    float k2_x = vx2, k2_y = vy2, k2_z = vz2;

    // RK4 Stage 3
    float vx3 = vx + hdt*k2_vx, vy3 = vy + hdt*k2_vy, vz3 = vz + hdt*k2_vz;
    float z3 = z + hdt*k2_z;
    float alt3 = std::max(z3, 0.0f);
    float gr3 = CPU_EARTH_RADIUS / (CPU_EARTH_RADIUS + alt3);
    float g3 = CPU_GRAVITY_G0 * gr3 * gr3;
    float rho3 = CPU_ATM_RHO0 * std::exp(-alt3 / CPU_ATM_SCALE_HEIGHT);
    float spd3 = std::sqrt(vx3*vx3 + vy3*vy3 + vz3*vz3);
    float df3 = CPU_BALLISTIC_BETA * rho3 * spd3;
    float k3_vx = -df3*vx3, k3_vy = -df3*vy3, k3_vz = -g3 - df3*vz3;
    float k3_x = vx3, k3_y = vy3, k3_z = vz3;

    // RK4 Stage 4
    float vx4 = vx + dt*k3_vx, vy4 = vy + dt*k3_vy, vz4 = vz + dt*k3_vz;
    float z4 = z + dt*k3_z;
    float alt4 = std::max(z4, 0.0f);
    float gr4 = CPU_EARTH_RADIUS / (CPU_EARTH_RADIUS + alt4);
    float g4 = CPU_GRAVITY_G0 * gr4 * gr4;
    float rho4 = CPU_ATM_RHO0 * std::exp(-alt4 / CPU_ATM_SCALE_HEIGHT);
    float spd4 = std::sqrt(vx4*vx4 + vy4*vy4 + vz4*vz4);
    float df4 = CPU_BALLISTIC_BETA * rho4 * spd4;
    float k4_vx = -df4*vx4, k4_vy = -df4*vy4, k4_vz = -g4 - df4*vz4;
    float k4_x = vx4, k4_y = vy4, k4_z = vz4;

    // RK4 結合
    float dt6 = dt / 6.0f;
    StateVector pred;
    pred(0) = x + dt6 * (k1_x + 2.0f*k2_x + 2.0f*k3_x + k4_x);
    pred(1) = y + dt6 * (k1_y + 2.0f*k2_y + 2.0f*k3_y + k4_y);
    pred(2) = z + dt6 * (k1_z + 2.0f*k2_z + 2.0f*k3_z + k4_z);
    pred(3) = vx + dt6 * (k1_vx + 2.0f*k2_vx + 2.0f*k3_vx + k4_vx);
    pred(4) = vy + dt6 * (k1_vy + 2.0f*k2_vy + 2.0f*k3_vy + k4_vy);
    pred(5) = vz + dt6 * (k1_vz + 2.0f*k2_vz + 2.0f*k3_vz + k4_vz);

    // 加速度: 最終状態から物理加速度を計算
    float alt_f = std::max(pred(2), 0.0f);
    float gr_f = CPU_EARTH_RADIUS / (CPU_EARTH_RADIUS + alt_f);
    float g_f = CPU_GRAVITY_G0 * gr_f * gr_f;
    float rho_f = CPU_ATM_RHO0 * std::exp(-alt_f / CPU_ATM_SCALE_HEIGHT);
    float spd_f = std::sqrt(pred(3)*pred(3) + pred(4)*pred(4) + pred(5)*pred(5));
    float df_f = CPU_BALLISTIC_BETA * rho_f * spd_f;
    pred(6) = -df_f * pred(3);
    pred(7) = -df_f * pred(4);
    pred(8) = -g_f - df_f * pred(5);
    return pred;
}

StateVector IMMFilter::predictCT_CPU(const StateVector& state, float dt) {
    float x = state(0), y = state(1), z = state(2);
    float vx = state(3), vy = state(4), vz = state(5);
    float ax = state(6), ay = state(7), az = state(8);

    StateVector pred;

    // 水平速度と旋回率
    float horiz_speed_sq = vx*vx + vy*vy;
    const float eps = 1e-3f;

    float omega = 0.0f;
    if (horiz_speed_sq > eps * eps) {
        omega = (vx * ay - vy * ax) / horiz_speed_sq;
    }
    // 旋回率クランプ（最大45 deg/s）
    omega = std::max(std::min(omega, 0.785f), -0.785f);

    if (std::abs(omega) > 1e-4f) {
        float sin_wt = std::sin(omega * dt);
        float cos_wt = std::cos(omega * dt);

        // 水平面: 旋回方程式
        pred(0) = x + (vx * sin_wt + vy * (cos_wt - 1.0f)) / omega;
        pred(1) = y + (-vx * (cos_wt - 1.0f) + vy * sin_wt) / omega;

        float vx_new = vx * cos_wt - vy * sin_wt;
        float vy_new = vx * sin_wt + vy * cos_wt;
        pred(3) = vx_new;
        pred(4) = vy_new;

        // 向心加速度
        pred(6) = -omega * vy_new;
        pred(7) = omega * vx_new;
    } else {
        // 直線近似（CA）
        float dt2 = 0.5f * dt * dt;
        pred(0) = x + vx * dt + ax * dt2;
        pred(1) = y + vy * dt + ay * dt2;
        pred(3) = vx + ax * dt;
        pred(4) = vy + ay * dt;
        pred(6) = ax;
        pred(7) = ay;
    }

    // 垂直方向: 等加速度
    pred(2) = z + vz * dt + 0.5f * az * dt * dt;
    pred(5) = vz + az * dt;
    pred(8) = az;
    return pred;
}

StateCov IMMFilter::computeApproxF(float dt) {
    // 簡易線形化ヤコビアン（等速直線ベース）
    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    StateCov F = StateCov::Identity();
    // 位置 ← 速度
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    return F;
}

// ================================================================
// IMM予測: 全モデル並行予測 + 重み付き統合
// ================================================================

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
    predicted_states.resize(num_targets);
    predicted_covs.resize(num_targets);
    updated_probs.resize(num_targets * num_models_);

    // モデル別予測を保存（尤度更新用）
    per_model_predictions_.resize(num_targets);
    per_model_pred_covs_.resize(num_targets);

    // 近似ヤコビアン（全モデル共通、共分散伝播用）
    StateCov F = computeApproxF(dt);

    for (int i = 0; i < num_targets; i++) {
        // -------------------------------------------------------
        // 1. 予測モデル確率 c_j = Σ_k π(k→j) * μ_k
        // -------------------------------------------------------
        float c[3] = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < num_models_; j++) {
            for (int k = 0; k < num_models_; k++) {
                int k_idx = i * num_models_ + k;
                if (k_idx < static_cast<int>(model_probs.size())) {
                    c[j] += transition_matrix_[k][j] * model_probs[k_idx];
                }
            }
            c[j] = std::max(c[j], 1e-10f);
        }

        // 正規化
        float c_sum = c[0] + c[1] + c[2];
        for (int j = 0; j < num_models_; j++) {
            c[j] /= c_sum;
        }

        // -------------------------------------------------------
        // 2. 各モデルで状態予測
        // -------------------------------------------------------
        StateVector x_pred[3];
        x_pred[0] = predictCV_CPU(states[i], dt);
        x_pred[1] = predictBallistic_CPU(states[i], dt);
        x_pred[2] = predictCT_CPU(states[i], dt);

        // モデル別予測を保存
        per_model_predictions_[i] = {x_pred[0], x_pred[1], x_pred[2]};

        // -------------------------------------------------------
        // 3. 各モデルの予測共分散 (線形近似 + モデル別Q)
        // -------------------------------------------------------
        StateCov P_pred[3];
        StateCov P_base = F * covariances[i] * F.transpose();

        for (int j = 0; j < num_models_; j++) {
            P_pred[j] = P_base;

            const auto& noise = process_noises_[j];
            float pos_var = noise.position_noise * noise.position_noise * dt;
            float vel_var = noise.velocity_noise * noise.velocity_noise * dt;
            float acc_var = noise.accel_noise * noise.accel_noise * dt;

            P_pred[j](0, 0) += pos_var;
            P_pred[j](1, 1) += pos_var;
            P_pred[j](2, 2) += pos_var;
            P_pred[j](3, 3) += vel_var;
            P_pred[j](4, 4) += vel_var;
            P_pred[j](5, 5) += vel_var;
            P_pred[j](6, 6) += acc_var;
            P_pred[j](7, 7) += acc_var;
            P_pred[j](8, 8) += acc_var;
        }

        // モデル別予測共分散を保存
        per_model_pred_covs_[i] = {P_pred[0], P_pred[1], P_pred[2]};

        // -------------------------------------------------------
        // 4. 重み付き統合
        //    x̂ = Σ c_j * x_pred_j
        //    P̂ = Σ c_j * [P_pred_j + (x_pred_j - x̂)(x_pred_j - x̂)']
        // -------------------------------------------------------
        StateVector x_combined = StateVector::Zero();
        for (int j = 0; j < num_models_; j++) {
            x_combined += c[j] * x_pred[j];
        }

        StateCov P_combined = StateCov::Zero();
        for (int j = 0; j < num_models_; j++) {
            StateVector diff = x_pred[j] - x_combined;
            P_combined += c[j] * (P_pred[j] + diff * diff.transpose());
        }

        predicted_states[i] = x_combined;
        predicted_covs[i] = P_combined;

        // -------------------------------------------------------
        // 5. 更新済みモデル確率（予測段階）
        // -------------------------------------------------------
        for (int j = 0; j < num_models_; j++) {
            updated_probs[i * num_models_ + j] = c[j];
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
    // 実際のKalman更新はGPU UKFで実行される
    // ここではモデル確率をパススルー
    updated_states = predicted_states;
    updated_covs = predicted_covs;
    updated_probs = model_probs;
}

// ================================================================
// 状態→観測変換（CPU版、尤度計算用）
// ================================================================
MeasVector IMMFilter::stateToMeas_CPU(const StateVector& state,
                                       float sensor_x, float sensor_y, float sensor_z) {
    float dx = state(0) - sensor_x;
    float dy = state(1) - sensor_y;
    float dz = state(2) - sensor_z;
    float vx = state(3), vy = state(4), vz = state(5);

    float r_horiz = std::sqrt(dx * dx + dy * dy);
    float range = std::sqrt(dx * dx + dy * dy + dz * dz);

    MeasVector z_pred;
    z_pred(0) = range;
    z_pred(1) = std::atan2(dy, dx);
    z_pred(2) = (r_horiz > 1e-6f) ? std::atan2(dz, r_horiz) : 0.0f;
    z_pred(3) = (range > 1e-6f) ? (dx * vx + dy * vy + dz * vz) / range : 0.0f;
    return z_pred;
}

// ================================================================
// 観測尤度ベースのモデル確率更新
//
// 正規IMMサイクル:
//   L_j = N(innovation_j; 0, S_j)
//   μ_j = c_j * L_j / Σ c_k * L_k
// ================================================================
void IMMFilter::updateModelProbabilities(
    const std::vector<int>& track_indices,
    const std::vector<Measurement>& measurements,
    std::vector<float>& model_probs,
    float sensor_x, float sensor_y, float sensor_z)
{
    // 観測ノイズ共分散 R (対角)
    MeasCov R = MeasCov::Zero();
    R(0, 0) = meas_noise_.range_noise * meas_noise_.range_noise;
    R(1, 1) = meas_noise_.azimuth_noise * meas_noise_.azimuth_noise;
    R(2, 2) = meas_noise_.elevation_noise * meas_noise_.elevation_noise;
    R(3, 3) = meas_noise_.doppler_noise * meas_noise_.doppler_noise;

    for (size_t k = 0; k < track_indices.size(); k++) {
        int i = track_indices[k];

        // 範囲チェック
        if (i < 0 || i >= static_cast<int>(per_model_predictions_.size())) continue;
        int prob_base = i * num_models_;
        if (prob_base + num_models_ > static_cast<int>(model_probs.size())) continue;

        // 実観測ベクトル
        MeasVector z_actual;
        z_actual(0) = measurements[k].range;
        z_actual(1) = measurements[k].azimuth;
        z_actual(2) = measurements[k].elevation;
        z_actual(3) = measurements[k].doppler;

        // 各モデルの尤度を計算
        float log_likelihoods[3];
        float max_log_lik = -1e30f;

        for (int j = 0; j < num_models_; j++) {
            // モデルjの予測状態から予測観測を計算
            MeasVector z_pred = stateToMeas_CPU(per_model_predictions_[i][j],
                                                 sensor_x, sensor_y, sensor_z);

            // イノベーション
            MeasVector innov = z_actual - z_pred;

            // 角度正規化
            while (innov(1) > static_cast<float>(M_PI)) innov(1) -= 2.0f * static_cast<float>(M_PI);
            while (innov(1) < -static_cast<float>(M_PI)) innov(1) += 2.0f * static_cast<float>(M_PI);
            while (innov(2) > static_cast<float>(M_PI)) innov(2) -= 2.0f * static_cast<float>(M_PI);
            while (innov(2) < -static_cast<float>(M_PI)) innov(2) += 2.0f * static_cast<float>(M_PI);

            // イノベーション共分散 S_j ≈ H * P_pred_j * H^T + R
            // 簡易: 観測モデルの線形近似ヤコビアンHは状態依存で複雑なため、
            // 予測共分散の位置/速度成分と R の和で近似
            MeasCov S = R;

            // 距離方向の不確かさ（位置共分散の射影）
            const auto& Pj = per_model_pred_covs_[i][j];
            float pos_var = (Pj(0, 0) + Pj(1, 1) + Pj(2, 2)) / 3.0f;
            float vel_var = (Pj(3, 3) + Pj(4, 4) + Pj(5, 5)) / 3.0f;
            float range_val = std::max(z_pred(0), 1000.0f);

            S(0, 0) += pos_var;  // range不確かさ
            S(1, 1) += pos_var / (range_val * range_val);  // azimuth不確かさ
            S(2, 2) += pos_var / (range_val * range_val);  // elevation不確かさ
            S(3, 3) += vel_var;  // doppler不確かさ

            // 対数尤度: -0.5 * (innovation^T * S^{-1} * innovation + log|S|)
            MeasCov S_inv = S.inverse();
            float mahal_sq = (innov.transpose() * S_inv * innov)(0);
            float log_det = std::log(std::max(S.determinant(), 1e-30f));

            log_likelihoods[j] = -0.5f * (mahal_sq + log_det);
            if (log_likelihoods[j] > max_log_lik) {
                max_log_lik = log_likelihoods[j];
            }
        }

        // ベイズ更新: μ_j = c_j * L_j / Σ c_k * L_k
        // 数値安定性のため log空間で正規化
        float weighted_sum = 0.0f;
        float weighted[3];
        for (int j = 0; j < num_models_; j++) {
            float c_j = model_probs[prob_base + j];
            // exp(log_lik - max_log_lik) で数値安定化
            float lik = std::exp(log_likelihoods[j] - max_log_lik);
            weighted[j] = c_j * lik;
            weighted_sum += weighted[j];
        }

        // 正規化（最小確率を下限で保証）
        const float min_prob = 0.01f;
        if (weighted_sum > 1e-30f) {
            for (int j = 0; j < num_models_; j++) {
                model_probs[prob_base + j] = std::max(weighted[j] / weighted_sum, min_prob);
            }
            // 再正規化
            float sum = model_probs[prob_base] + model_probs[prob_base + 1] + model_probs[prob_base + 2];
            for (int j = 0; j < num_models_; j++) {
                model_probs[prob_base + j] /= sum;
            }
        }
    }
}

} // namespace fasttracker
