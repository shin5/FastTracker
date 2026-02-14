#include "ukf/ukf_kernels.cuh"
#include "utils/matrix.cuh"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {
namespace cuda {

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

    // 状態とオフセット
    const float* state = &states[tid * n];
    const float* cov = &covariances[tid * n * n];
    float* sigma = &sigma_points[tid * SIGMA_POINTS * n];

    // Cholesky分解: cov = L * L^T
    float L[STATE_DIM * STATE_DIM];
    bool success = cholesky(cov, L, n);

    if (!success) {
        // Cholesky分解失敗時は単位行列を使用（フォールバック）
        for (int i = 0; i < n * n; i++) L[i] = 0.0f;
        for (int i = 0; i < n; i++) L[i * n + i] = 1.0f;
    }

    // シグマポイント0: 平均
    for (int i = 0; i < n; i++) {
        sigma[i] = state[i];
    }

    // シグマポイント 1~n: state + scale * L[:, i]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sigma[(i + 1) * n + j] = state[j] + scale * L[j * n + i];
        }
    }

    // シグマポイント n+1~2n: state - scale * L[:, i]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sigma[(i + n + 1) * n + j] = state[j] - scale * L[j * n + i];
        }
    }
}

__global__ void predictSigmaPoints(
    const float* sigma_points,
    float* predicted_sigma_points,
    int num_targets,
    float dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = num_targets * SIGMA_POINTS;
    if (tid >= total_points) return;

    const int n = STATE_DIM;
    const float* sp = &sigma_points[tid * n];
    float* pred_sp = &predicted_sigma_points[tid * n];

    // 状態: [x, y, vx, vy, ax, ay]
    float x = sp[0];
    float y = sp[1];
    float vx = sp[2];
    float vy = sp[3];
    float ax = sp[4];
    float ay = sp[5];

    // 等加速度運動モデル
    // x_new = x + vx*dt + 0.5*ax*dt^2
    // vx_new = vx + ax*dt
    float dt2 = 0.5f * dt * dt;

    pred_sp[0] = x + vx * dt + ax * dt2;
    pred_sp[1] = y + vy * dt + ay * dt2;
    pred_sp[2] = vx + ax * dt;
    pred_sp[3] = vy + ay * dt;
    pred_sp[4] = ax;  // 加速度は一定
    pred_sp[5] = ay;
}

__global__ void measurementModel(
    const float* sigma_points,
    float* meas_sigma_points,
    int num_targets,
    float sensor_x,
    float sensor_y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = num_targets * SIGMA_POINTS;
    if (tid >= total_points) return;

    const int n = STATE_DIM;
    const int m = MEAS_DIM;
    const float* sp = &sigma_points[tid * n];
    float* meas = &meas_sigma_points[tid * m];

    // 状態: [x, y, vx, vy, ax, ay]
    // センサーからの相対座標で観測量を計算
    float dx = sp[0] - sensor_x;
    float dy = sp[1] - sensor_y;
    float vx = sp[2];
    float vy = sp[3];

    // レーダー観測モデル（センサー位置基準）
    float range = sqrtf(dx * dx + dy * dy);
    float azimuth = atan2f(dy, dx);
    float elevation = 0.0f;  // 2D追尾では0

    // ドップラー速度: 視線方向の速度成分
    float doppler = 0.0f;
    if (range > 1e-6f) {
        doppler = (dx * vx + dy * vy) / range;
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

    // 初期化
    for (int i = 0; i < dim; i++) {
        m[i] = 0.0f;
    }

    // 重み付き和
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

    // 初期化
    for (int i = 0; i < dim * dim; i++) {
        cov[i] = 0.0f;
    }

    // Cov = Σ w_k * (X_k - μ) * (X_k - μ)^T
    for (int k = 0; k < SIGMA_POINTS; k++) {
        float w = weights_cov[k];
        const float* sp_k = &sp[k * dim];

        // 差分ベクトル
        float diff[STATE_DIM];  // 最大次元
        for (int i = 0; i < dim; i++) {
            diff[i] = sp_k[i] - m[i];
        }

        // 外積: diff * diff^T
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

    // 初期化
    for (int i = 0; i < n * m; i++) {
        cross[i] = 0.0f;
    }

    // P_xy = Σ w_k * (X_k - μ_x) * (Y_k - μ_y)^T
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

        // 外積
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
    int dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_targets) return;

    float* cov = &covariance[tid * dim * dim];

    for (int i = 0; i < dim * dim; i++) {
        cov[i] += noise_cov[i];
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

    // S の逆行列を計算
    float S_inv[MEAS_DIM * MEAS_DIM];
    bool success = invert(S, S_inv, m);

    if (!success) {
        // 逆行列計算失敗時はゼロゲインを使用
        for (int i = 0; i < n * m; i++) {
            K[i] = 0.0f;
        }
        return;
    }

    // K = P_xy * S^{-1}
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

    // イノベーション: y = z - z_pred
    float innovation[MEAS_DIM];
    for (int i = 0; i < m; i++) {
        innovation[i] = z[i] - z_pred[i];
    }

    // 角度の正規化（方位角）
    // -π ~ π に正規化
    if (innovation[1] > M_PI) {
        innovation[1] -= 2.0f * M_PI;
    } else if (innovation[1] < -M_PI) {
        innovation[1] += 2.0f * M_PI;
    }

    // 状態更新: x = x + K * innovation
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

    // Joseph form: P = P - K * S * K^T
    // K * S
    float KS[STATE_DIM * MEAS_DIM];
    matmul(K, S, KS, n, m, m);

    // (K * S) * K^T
    float K_T[MEAS_DIM * STATE_DIM];
    transpose(K, K_T, n, m);

    float KSK_T[STATE_DIM * STATE_DIM];
    matmul(KS, K_T, KSK_T, n, m, n);

    // P = P - KSK_T
    for (int i = 0; i < n * n; i++) {
        P[i] -= KSK_T[i];
    }

    // 対称性を強制（数値安定性のため）
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
    float lambda)
{
    // 融合カーネル: 1ブロックで1目標を処理
    // 各スレッドがシグマポイントを担当

    int target_id = blockIdx.x;
    if (target_id >= num_targets) return;

    int tid = threadIdx.x;
    const int n = STATE_DIM;

    // 共有メモリ（シグマポイント、平均、共分散の一時保存）
    __shared__ float s_sigma_points[SIGMA_POINTS * STATE_DIM];
    __shared__ float s_pred_sigma_points[SIGMA_POINTS * STATE_DIM];
    __shared__ float s_mean[STATE_DIM];
    __shared__ float s_cov[STATE_DIM * STATE_DIM];

    // 状態と共分散の読み込み
    float* state = &states[target_id * n];
    float* cov = &covariances[target_id * n * n];

    // 1. シグマポイント生成（1スレッドで実行）
    if (tid == 0) {
        const float scale = sqrtf(n + lambda);
        float L[STATE_DIM * STATE_DIM];
        cholesky(cov, L, n);

        // シグマポイント0
        for (int i = 0; i < n; i++) {
            s_sigma_points[i] = state[i];
        }

        // シグマポイント1~2n
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s_sigma_points[(i + 1) * n + j] = state[j] + scale * L[j * n + i];
                s_sigma_points[(i + n + 1) * n + j] = state[j] - scale * L[j * n + i];
            }
        }
    }
    __syncthreads();

    // 2. シグマポイント予測（各スレッドが1つのシグマポイントを処理）
    if (tid < SIGMA_POINTS) {
        const float* sp = &s_sigma_points[tid * n];
        float* pred_sp = &s_pred_sigma_points[tid * n];

        float x = sp[0], y = sp[1];
        float vx = sp[2], vy = sp[3];
        float ax = sp[4], ay = sp[5];

        float dt2 = 0.5f * dt * dt;

        pred_sp[0] = x + vx * dt + ax * dt2;
        pred_sp[1] = y + vy * dt + ay * dt2;
        pred_sp[2] = vx + ax * dt;
        pred_sp[3] = vy + ay * dt;
        pred_sp[4] = ax;
        pred_sp[5] = ay;
    }
    __syncthreads();

    // 3. 平均計算（並列リダクション）
    if (tid < n) {
        float sum = 0.0f;
        for (int k = 0; k < SIGMA_POINTS; k++) {
            sum += weights_mean[k] * s_pred_sigma_points[k * n + tid];
        }
        s_mean[tid] = sum;
    }
    __syncthreads();

    // 4. 共分散計算（各スレッドが行列の一部を担当）
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
        cov_ij += process_cov[i * n + j];  // プロセスノイズ加算
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
