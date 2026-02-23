#include "ukf/ukf_kernels.cuh"
#include "utils/matrix.cuh"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {
namespace cuda {

// 物理定数（弾道モデル用）
__constant__ float GRAVITY_G0 = 9.80665f;
__constant__ float EARTH_RADIUS = 6371000.0f;
__constant__ float ATM_RHO0 = 1.225f;
__constant__ float ATM_SCALE_HEIGHT = 7400.0f;
__constant__ float BALLISTIC_BETA = 0.001f;  // Cd*A/(2*m) 代表値

// ================================================================
// 3つの運動モデル (__device__ 関数)
// 状態: [x, y, z, vx, vy, vz, ax, ay, az]
// ================================================================

__device__ void predictCV(const float* sp, float* pred_sp, float dt) {
    float x = sp[0], y = sp[1], z = sp[2];
    float vx = sp[3], vy = sp[4], vz = sp[5];

    // 位置: 等速直線
    pred_sp[0] = x + vx * dt;
    pred_sp[1] = y + vy * dt;
    pred_sp[2] = z + vz * dt;

    // 速度: 一定
    pred_sp[3] = vx;
    pred_sp[4] = vy;
    pred_sp[5] = vz;

    // 加速度: 減衰 (τ=5s)
    float decay = expf(-dt / 5.0f);
    pred_sp[6] = sp[6] * decay;
    pred_sp[7] = sp[7] * decay;
    pred_sp[8] = sp[8] * decay;
}

__device__ void predictBallistic(const float* sp, float* pred_sp, float dt) {
    float x = sp[0], y = sp[1], z = sp[2];
    float vx = sp[3], vy = sp[4], vz = sp[5];

    // --- RK4 積分 ---
    // Stage 1
    float alt1 = fmaxf(z, 0.0f);
    float gr1 = EARTH_RADIUS / (EARTH_RADIUS + alt1);
    float g1 = GRAVITY_G0 * gr1 * gr1;
    float rho1 = ATM_RHO0 * expf(-alt1 / ATM_SCALE_HEIGHT);
    float spd1 = sqrtf(vx*vx + vy*vy + vz*vz);
    float df1 = BALLISTIC_BETA * rho1 * spd1;
    float k1_vx = -df1*vx, k1_vy = -df1*vy, k1_vz = -g1 - df1*vz;
    float k1_x = vx, k1_y = vy, k1_z = vz;

    // Stage 2
    float hdt = 0.5f * dt;
    float vx2 = vx + hdt*k1_vx, vy2 = vy + hdt*k1_vy, vz2 = vz + hdt*k1_vz;
    float z2 = z + hdt*k1_z;
    float alt2 = fmaxf(z2, 0.0f);
    float gr2 = EARTH_RADIUS / (EARTH_RADIUS + alt2);
    float g2 = GRAVITY_G0 * gr2 * gr2;
    float rho2 = ATM_RHO0 * expf(-alt2 / ATM_SCALE_HEIGHT);
    float spd2 = sqrtf(vx2*vx2 + vy2*vy2 + vz2*vz2);
    float df2 = BALLISTIC_BETA * rho2 * spd2;
    float k2_vx = -df2*vx2, k2_vy = -df2*vy2, k2_vz = -g2 - df2*vz2;
    float k2_x = vx2, k2_y = vy2, k2_z = vz2;

    // Stage 3
    float vx3 = vx + hdt*k2_vx, vy3 = vy + hdt*k2_vy, vz3 = vz + hdt*k2_vz;
    float z3 = z + hdt*k2_z;
    float alt3 = fmaxf(z3, 0.0f);
    float gr3 = EARTH_RADIUS / (EARTH_RADIUS + alt3);
    float g3 = GRAVITY_G0 * gr3 * gr3;
    float rho3 = ATM_RHO0 * expf(-alt3 / ATM_SCALE_HEIGHT);
    float spd3 = sqrtf(vx3*vx3 + vy3*vy3 + vz3*vz3);
    float df3 = BALLISTIC_BETA * rho3 * spd3;
    float k3_vx = -df3*vx3, k3_vy = -df3*vy3, k3_vz = -g3 - df3*vz3;
    float k3_x = vx3, k3_y = vy3, k3_z = vz3;

    // Stage 4
    float vx4 = vx + dt*k3_vx, vy4 = vy + dt*k3_vy, vz4 = vz + dt*k3_vz;
    float z4 = z + dt*k3_z;
    float alt4 = fmaxf(z4, 0.0f);
    float gr4 = EARTH_RADIUS / (EARTH_RADIUS + alt4);
    float g4 = GRAVITY_G0 * gr4 * gr4;
    float rho4 = ATM_RHO0 * expf(-alt4 / ATM_SCALE_HEIGHT);
    float spd4 = sqrtf(vx4*vx4 + vy4*vy4 + vz4*vz4);
    float df4 = BALLISTIC_BETA * rho4 * spd4;
    float k4_vx = -df4*vx4, k4_vy = -df4*vy4, k4_vz = -g4 - df4*vz4;
    float k4_x = vx4, k4_y = vy4, k4_z = vz4;

    // RK4 結合
    float dt6 = dt / 6.0f;
    pred_sp[0] = x + dt6 * (k1_x + 2.0f*k2_x + 2.0f*k3_x + k4_x);
    pred_sp[1] = y + dt6 * (k1_y + 2.0f*k2_y + 2.0f*k3_y + k4_y);
    pred_sp[2] = z + dt6 * (k1_z + 2.0f*k2_z + 2.0f*k3_z + k4_z);
    pred_sp[3] = vx + dt6 * (k1_vx + 2.0f*k2_vx + 2.0f*k3_vx + k4_vx);
    pred_sp[4] = vy + dt6 * (k1_vy + 2.0f*k2_vy + 2.0f*k3_vy + k4_vy);
    pred_sp[5] = vz + dt6 * (k1_vz + 2.0f*k2_vz + 2.0f*k3_vz + k4_vz);

    // 加速度: 最終状態から物理加速度を計算
    float alt_f = fmaxf(pred_sp[2], 0.0f);
    float gr_f = EARTH_RADIUS / (EARTH_RADIUS + alt_f);
    float g_f = GRAVITY_G0 * gr_f * gr_f;
    float rho_f = ATM_RHO0 * expf(-alt_f / ATM_SCALE_HEIGHT);
    float spd_f = sqrtf(pred_sp[3]*pred_sp[3] + pred_sp[4]*pred_sp[4] + pred_sp[5]*pred_sp[5]);
    float df_f = BALLISTIC_BETA * rho_f * spd_f;
    pred_sp[6] = -df_f * pred_sp[3];
    pred_sp[7] = -df_f * pred_sp[4];
    pred_sp[8] = -g_f - df_f * pred_sp[5];
}

__device__ void predictCoordinatedTurn(const float* sp, float* pred_sp, float dt) {
    float x = sp[0], y = sp[1], z = sp[2];
    float vx = sp[3], vy = sp[4], vz = sp[5];
    float ax = sp[6], ay = sp[7], az = sp[8];

    // 水平速度と旋回率
    float horiz_speed_sq = vx*vx + vy*vy;
    const float eps = 1e-3f;

    float omega = 0.0f;
    if (horiz_speed_sq > eps * eps) {
        omega = (vx * ay - vy * ax) / horiz_speed_sq;
    }
    // 旋回率クランプ（最大45 deg/s）
    omega = fmaxf(fminf(omega, 0.785f), -0.785f);

    if (fabsf(omega) > 1e-4f) {
        float sin_wt = sinf(omega * dt);
        float cos_wt = cosf(omega * dt);

        // 水平面: 旋回方程式
        pred_sp[0] = x + (vx * sin_wt + vy * (cos_wt - 1.0f)) / omega;
        pred_sp[1] = y + (-vx * (cos_wt - 1.0f) + vy * sin_wt) / omega;

        float vx_new = vx * cos_wt - vy * sin_wt;
        float vy_new = vx * sin_wt + vy * cos_wt;
        pred_sp[3] = vx_new;
        pred_sp[4] = vy_new;

        // 向心加速度
        pred_sp[6] = -omega * vy_new;
        pred_sp[7] = omega * vx_new;
    } else {
        // 直線近似（CA）
        float dt2 = 0.5f * dt * dt;
        pred_sp[0] = x + vx * dt + ax * dt2;
        pred_sp[1] = y + vy * dt + ay * dt2;
        pred_sp[3] = vx + ax * dt;
        pred_sp[4] = vy + ay * dt;
        pred_sp[6] = ax;
        pred_sp[7] = ay;
    }

    // 垂直方向: 等加速度 + 重力
    pred_sp[2] = z + vz * dt + 0.5f * az * dt * dt;
    pred_sp[5] = vz + az * dt;
    pred_sp[8] = az;
}

// ================================================================
// カーネル実装
// ================================================================

__global__ void generateSigmaPoints(
    const float* states,
    const float* covariances,
    float* sigma_points,
    int num_targets,
    float lambda)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    const int n = STATE_DIM;
    const float scale = sqrtf(n + lambda);

    const float* state = &states[tid * n];
    const float* cov = &covariances[tid * n * n];
    float* sigma = &sigma_points[tid * SIGMA_POINTS * n];

    // Cholesky分解
    float L[STATE_DIM * STATE_DIM];
    bool success = cholesky(cov, L, n);

    if (!success) {
        for (int i = 0; i < n * n; i++) L[i] = 0.0f;
        for (int i = 0; i < n; i++) L[i * n + i] = 1.0f;
    }

    // シグマポイント0: 平均
    for (int i = 0; i < n; i++) {
        sigma[i] = state[i];
    }

    // シグマポイント 1~2n
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sigma[(i + 1) * n + j] = state[j] + scale * L[j * n + i];
            sigma[(i + n + 1) * n + j] = state[j] - scale * L[j * n + i];
        }
    }
}

__global__ void predictSigmaPoints(
    const float* sigma_points,
    float* predicted_sigma_points,
    int num_targets,
    float dt,
    int model_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = num_targets * SIGMA_POINTS;
    if (tid >= total_points) return;

    const int n = STATE_DIM;
    const float* sp = &sigma_points[tid * n];
    float* pred_sp = &predicted_sigma_points[tid * n];

    switch (model_id) {
        case 0: predictCV(sp, pred_sp, dt); break;
        case 1: predictBallistic(sp, pred_sp, dt); break;
        case 2: predictCoordinatedTurn(sp, pred_sp, dt); break;
        default: predictCV(sp, pred_sp, dt); break;
    }
}

__global__ void measurementModel(
    const float* sigma_points,
    float* meas_sigma_points,
    int num_targets,
    float sensor_x,
    float sensor_y,
    float sensor_z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = num_targets * SIGMA_POINTS;
    if (tid >= total_points) return;

    const int n = STATE_DIM;
    const int m = MEAS_DIM;
    const float* sp = &sigma_points[tid * n];
    float* meas = &meas_sigma_points[tid * m];

    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    float dx = sp[0] - sensor_x;
    float dy = sp[1] - sensor_y;
    float dz = sp[2] - sensor_z;
    float vx = sp[3];
    float vy = sp[4];
    float vz = sp[5];

    float range_horiz = sqrtf(dx * dx + dy * dy);
    float range = sqrtf(dx * dx + dy * dy + dz * dz);
    float azimuth = atan2f(dy, dx);
    float elevation = (range_horiz > 1e-6f) ? atan2f(dz, range_horiz) : 0.0f;

    float doppler = 0.0f;
    if (range > 1e-6f) {
        doppler = (dx * vx + dy * vy + dz * vz) / range;
    }

    meas[0] = range;
    meas[1] = azimuth;
    meas[2] = elevation;
    meas[3] = doppler;
}

__global__ void computeWeightedMean(
    const float* sigma_points,
    const float* weights,
    float* mean,
    int num_targets,
    int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    const float* sp = &sigma_points[tid * SIGMA_POINTS * dim];
    float* m = &mean[tid * dim];

    for (int i = 0; i < dim; i++) {
        m[i] = 0.0f;
    }

    for (int k = 0; k < SIGMA_POINTS; k++) {
        float w = weights[k];
        for (int i = 0; i < dim; i++) {
            m[i] += w * sp[k * dim + i];
        }
    }
}

__global__ void computeCovariance(
    const float* sigma_points,
    const float* mean,
    const float* weights_cov,
    float* covariance,
    int num_targets,
    int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    const float* sp = &sigma_points[tid * SIGMA_POINTS * dim];
    const float* m = &mean[tid * dim];
    float* cov = &covariance[tid * dim * dim];

    for (int i = 0; i < dim * dim; i++) {
        cov[i] = 0.0f;
    }

    for (int k = 0; k < SIGMA_POINTS; k++) {
        float w = weights_cov[k];
        const float* sp_k = &sp[k * dim];

        float diff[STATE_DIM];  // 最大次元
        for (int i = 0; i < dim; i++) {
            diff[i] = sp_k[i] - m[i];
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                cov[i * dim + j] += w * diff[i] * diff[j];
            }
        }
    }
}

__global__ void computeCrossCov(
    const float* state_sigma_points,
    const float* meas_sigma_points,
    const float* state_mean,
    const float* meas_mean,
    const float* weights_cov,
    float* cross_cov,
    int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    const int n = STATE_DIM;
    const int m = MEAS_DIM;

    const float* state_sp = &state_sigma_points[tid * SIGMA_POINTS * n];
    const float* meas_sp = &meas_sigma_points[tid * SIGMA_POINTS * m];
    const float* state_m = &state_mean[tid * n];
    const float* meas_m = &meas_mean[tid * m];
    float* cross = &cross_cov[tid * n * m];

    for (int i = 0; i < n * m; i++) {
        cross[i] = 0.0f;
    }

    for (int k = 0; k < SIGMA_POINTS; k++) {
        float w = weights_cov[k];

        const float* state_k = &state_sp[k * n];
        const float* meas_k = &meas_sp[k * m];

        float state_diff[STATE_DIM];
        float meas_diff[MEAS_DIM];

        for (int i = 0; i < n; i++) {
            state_diff[i] = state_k[i] - state_m[i];
        }
        for (int i = 0; i < m; i++) {
            meas_diff[i] = meas_k[i] - meas_m[i];
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cross[i * m + j] += w * state_diff[i] * meas_diff[j];
            }
        }
    }
}

__global__ void addNoiseCov(
    float* covariance,
    const float* noise_cov,
    int num_targets,
    int dim,
    float dt_scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    float* cov = &covariance[tid * dim * dim];

    for (int i = 0; i < dim * dim; i++) {
        cov[i] += noise_cov[i] * dt_scale;
    }
}

__global__ void computeKalmanGain(
    const float* cross_cov,
    const float* innovation_cov,
    float* kalman_gain,
    int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    const int n = STATE_DIM;
    const int m = MEAS_DIM;

    const float* P_xy = &cross_cov[tid * n * m];
    const float* S = &innovation_cov[tid * m * m];
    float* K = &kalman_gain[tid * n * m];

    float S_inv[MEAS_DIM * MEAS_DIM];
    bool success = invert(S, S_inv, m);

    if (!success) {
        for (int i = 0; i < n * m; i++) {
            K[i] = 0.0f;
        }
        return;
    }

    matmul(P_xy, S_inv, K, n, m, m);
}

__global__ void updateState(
    float* states,
    const float* kalman_gain,
    const float* measurements,
    const float* pred_measurements,
    int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    const int n = STATE_DIM;
    const int m = MEAS_DIM;

    float* state = &states[tid * n];
    const float* K = &kalman_gain[tid * n * m];
    const float* z = &measurements[tid * m];
    const float* z_pred = &pred_measurements[tid * m];

    float innovation[MEAS_DIM];
    for (int i = 0; i < m; i++) {
        innovation[i] = z[i] - z_pred[i];
    }

    // 方位角の正規化
    if (innovation[1] > static_cast<float>(M_PI)) {
        innovation[1] -= 2.0f * static_cast<float>(M_PI);
    } else if (innovation[1] < -static_cast<float>(M_PI)) {
        innovation[1] += 2.0f * static_cast<float>(M_PI);
    }

    // 仰角の正規化
    if (innovation[2] > static_cast<float>(M_PI)) {
        innovation[2] -= 2.0f * static_cast<float>(M_PI);
    } else if (innovation[2] < -static_cast<float>(M_PI)) {
        innovation[2] += 2.0f * static_cast<float>(M_PI);
    }

    for (int i = 0; i < n; i++) {
        float update = 0.0f;
        for (int j = 0; j < m; j++) {
            update += K[i * m + j] * innovation[j];
        }
        state[i] += update;
    }
}

__global__ void updateCovariance(
    float* covariances,
    const float* kalman_gain,
    const float* innovation_cov,
    int num_targets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    const int n = STATE_DIM;
    const int m = MEAS_DIM;

    float* P = &covariances[tid * n * n];
    const float* K = &kalman_gain[tid * n * m];
    const float* S = &innovation_cov[tid * m * m];

    float KS[STATE_DIM * MEAS_DIM];
    matmul(K, S, KS, n, m, m);

    float K_T[MEAS_DIM * STATE_DIM];
    transpose(K, K_T, n, m);

    float KSK_T[STATE_DIM * STATE_DIM];
    matmul(KS, K_T, KSK_T, n, m, n);

    for (int i = 0; i < n * n; i++) {
        P[i] -= KSK_T[i];
    }

    // 対称性強制
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float avg = 0.5f * (P[i * n + j] + P[j * n + i]);
            P[i * n + j] = avg;
            P[j * n + i] = avg;
        }
    }
}

__global__ void fusedPredict(
    float* states,
    float* covariances,
    const float* process_cov,
    const float* weights_mean,
    const float* weights_cov,
    int num_targets,
    float dt,
    float lambda,
    int model_id)
{
    int target_id = blockIdx.x;
    if (target_id >= num_targets) return;

    int tid = threadIdx.x;
    const int n = STATE_DIM;

    __shared__ float s_sigma_points[SIGMA_POINTS * STATE_DIM];
    __shared__ float s_pred_sigma_points[SIGMA_POINTS * STATE_DIM];
    __shared__ float s_mean[STATE_DIM];
    __shared__ float s_cov[STATE_DIM * STATE_DIM];

    float* state = &states[target_id * n];
    float* cov = &covariances[target_id * n * n];

    // 1. シグマポイント生成（1スレッドで実行）
    if (tid == 0) {
        const float scale = sqrtf(n + lambda);
        float L[STATE_DIM * STATE_DIM];
        bool chol_ok = cholesky(cov, L, n);
        if (!chol_ok) {
            // Covariance is not positive-definite (numerical drift); use identity
            // to prevent NaN sigma points that would corrupt downstream calculations.
            for (int i = 0; i < n * n; i++) L[i] = 0.0f;
            for (int i = 0; i < n; i++) L[i * n + i] = 1.0f;
        }

        for (int i = 0; i < n; i++) {
            s_sigma_points[i] = state[i];
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s_sigma_points[(i + 1) * n + j] = state[j] + scale * L[j * n + i];
                s_sigma_points[(i + n + 1) * n + j] = state[j] - scale * L[j * n + i];
            }
        }
    }
    __syncthreads();

    // 2. シグマポイント予測（モデル別）
    if (tid < SIGMA_POINTS) {
        const float* sp = &s_sigma_points[tid * n];
        float* pred_sp = &s_pred_sigma_points[tid * n];

        switch (model_id) {
            case 0: predictCV(sp, pred_sp, dt); break;
            case 1: predictBallistic(sp, pred_sp, dt); break;
            case 2: predictCoordinatedTurn(sp, pred_sp, dt); break;
            default: predictCV(sp, pred_sp, dt); break;
        }
    }
    __syncthreads();

    // 3. 平均計算
    if (tid < n) {
        float sum = 0.0f;
        for (int k = 0; k < SIGMA_POINTS; k++) {
            sum += weights_mean[k] * s_pred_sigma_points[k * n + tid];
        }
        s_mean[tid] = sum;
    }
    __syncthreads();

    // 4. 共分散計算
    int cov_elements = n * n;
    for (int idx = tid; idx < cov_elements; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;

        float cov_ij = 0.0f;
        for (int k = 0; k < SIGMA_POINTS; k++) {
            float diff_i = s_pred_sigma_points[k * n + i] - s_mean[i];
            float diff_j = s_pred_sigma_points[k * n + j] - s_mean[j];
            cov_ij += weights_cov[k] * diff_i * diff_j;
        }
        cov_ij += process_cov[i * n + j] * dt;
        s_cov[i * n + j] = cov_ij;
    }
    __syncthreads();

    // 5. 結果を書き戻し
    if (tid < n) {
        state[tid] = s_mean[tid];
    }
    for (int idx = tid; idx < cov_elements; idx += blockDim.x) {
        cov[idx] = s_cov[idx];
    }
}

} // namespace cuda
} // namespace fasttracker
