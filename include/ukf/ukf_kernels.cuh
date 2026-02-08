#ifndef FASTTRACKER_UKF_KERNELS_CUH
#define FASTTRACKER_UKF_KERNELS_CUH

#include <cuda_runtime.h>
#include "utils/types.hpp"

namespace fasttracker {
namespace cuda {

/**
 * @brief シグマポイント生成カーネル
 *
 * 各目標に対してシグマポイントを生成します。
 * スレッドは目標ごとに割り当てられ、各スレッドが2n+1個のシグマポイントを生成します。
 *
 * @param states 状態ベクトル [num_targets * STATE_DIM]
 * @param covariances 共分散行列 [num_targets * STATE_DIM * STATE_DIM]
 * @param sigma_points 出力シグマポイント [num_targets * SIGMA_POINTS * STATE_DIM]
 * @param num_targets 目標数
 * @param lambda UKFスケーリングパラメータ
 */
__global__ void generateSigmaPoints(
    const float* states,
    const float* covariances,
    float* sigma_points,
    int num_targets,
    float lambda
);

/**
 * @brief 状態遷移関数（予測モデル）カーネル
 *
 * 等加速度運動モデル: x_{k+1} = F * x_k + w
 * 状態: [x, y, vx, vy, ax, ay]
 *
 * @param sigma_points 入力シグマポイント [num_targets * SIGMA_POINTS * STATE_DIM]
 * @param predicted_sigma_points 出力シグマポイント [num_targets * SIGMA_POINTS * STATE_DIM]
 * @param num_targets 目標数
 * @param dt 時間差分 [s]
 */
__global__ void predictSigmaPoints(
    const float* sigma_points,
    float* predicted_sigma_points,
    int num_targets,
    float dt
);

/**
 * @brief 観測モデルカーネル
 *
 * レーダー観測: [range, azimuth, elevation, doppler]
 * range = sqrt(x^2 + y^2)
 * azimuth = atan2(y, x)
 * elevation = atan2(z, sqrt(x^2 + y^2))  // 2D追尾では0
 * doppler = (x*vx + y*vy) / range
 *
 * @param sigma_points 状態シグマポイント [num_targets * SIGMA_POINTS * STATE_DIM]
 * @param meas_sigma_points 観測シグマポイント [num_targets * SIGMA_POINTS * MEAS_DIM]
 * @param num_targets 目標数
 */
__global__ void measurementModel(
    const float* sigma_points,
    float* meas_sigma_points,
    int num_targets
);

/**
 * @brief 重み付き平均計算カーネル
 *
 * 状態ベクトルまたは観測ベクトルの重み付き平均を計算します。
 *
 * @param sigma_points シグマポイント [num_targets * SIGMA_POINTS * dim]
 * @param weights 重み [SIGMA_POINTS]
 * @param mean 出力平均 [num_targets * dim]
 * @param num_targets 目標数
 * @param dim 次元数（STATE_DIMまたはMEAS_DIM）
 */
__global__ void computeWeightedMean(
    const float* sigma_points,
    const float* weights,
    float* mean,
    int num_targets,
    int dim
);

/**
 * @brief 共分散行列計算カーネル
 *
 * シグマポイントから共分散行列を計算します。
 * Cov = Σ w_i * (X_i - μ) * (X_i - μ)^T
 *
 * @param sigma_points シグマポイント [num_targets * SIGMA_POINTS * dim]
 * @param mean 平均 [num_targets * dim]
 * @param weights_cov 共分散重み [SIGMA_POINTS]
 * @param covariance 出力共分散 [num_targets * dim * dim]
 * @param num_targets 目標数
 * @param dim 次元数
 */
__global__ void computeCovariance(
    const float* sigma_points,
    const float* mean,
    const float* weights_cov,
    float* covariance,
    int num_targets,
    int dim
);

/**
 * @brief クロス共分散行列計算カーネル
 *
 * 状態と観測のクロス共分散を計算します。
 * P_xy = Σ w_i * (X_i - μ_x) * (Y_i - μ_y)^T
 *
 * @param state_sigma_points 状態シグマポイント [num_targets * SIGMA_POINTS * STATE_DIM]
 * @param meas_sigma_points 観測シグマポイント [num_targets * SIGMA_POINTS * MEAS_DIM]
 * @param state_mean 状態平均 [num_targets * STATE_DIM]
 * @param meas_mean 観測平均 [num_targets * MEAS_DIM]
 * @param weights_cov 共分散重み [SIGMA_POINTS]
 * @param cross_cov 出力クロス共分散 [num_targets * STATE_DIM * MEAS_DIM]
 * @param num_targets 目標数
 */
__global__ void computeCrossCov(
    const float* state_sigma_points,
    const float* meas_sigma_points,
    const float* state_mean,
    const float* meas_mean,
    const float* weights_cov,
    float* cross_cov,
    int num_targets
);

/**
 * @brief ノイズ共分散行列加算カーネル
 *
 * 予測共分散にプロセスノイズを加算、または
 * イノベーション共分散に観測ノイズを加算します。
 *
 * @param covariance 共分散行列 [num_targets * dim * dim]
 * @param noise_cov ノイズ共分散 [dim * dim]
 * @param num_targets 目標数
 * @param dim 次元数
 */
__global__ void addNoiseCov(
    float* covariance,
    const float* noise_cov,
    int num_targets,
    int dim
);

/**
 * @brief カルマンゲイン計算カーネル
 *
 * K = P_xy * S^{-1}
 * P_xy: クロス共分散 [STATE_DIM x MEAS_DIM]
 * S: イノベーション共分散 [MEAS_DIM x MEAS_DIM]
 * K: カルマンゲイン [STATE_DIM x MEAS_DIM]
 *
 * @param cross_cov クロス共分散 [num_targets * STATE_DIM * MEAS_DIM]
 * @param innovation_cov イノベーション共分散 [num_targets * MEAS_DIM * MEAS_DIM]
 * @param kalman_gain 出力カルマンゲイン [num_targets * STATE_DIM * MEAS_DIM]
 * @param num_targets 目標数
 */
__global__ void computeKalmanGain(
    const float* cross_cov,
    const float* innovation_cov,
    float* kalman_gain,
    int num_targets
);

/**
 * @brief 状態更新カーネル
 *
 * x = x + K * (z - z_pred)
 *
 * @param states 状態ベクトル [num_targets * STATE_DIM]（入出力）
 * @param kalman_gain カルマンゲイン [num_targets * STATE_DIM * MEAS_DIM]
 * @param measurements 実際の観測 [num_targets * MEAS_DIM]
 * @param pred_measurements 予測観測 [num_targets * MEAS_DIM]
 * @param num_targets 目標数
 */
__global__ void updateState(
    float* states,
    const float* kalman_gain,
    const float* measurements,
    const float* pred_measurements,
    int num_targets
);

/**
 * @brief 共分散更新カーネル
 *
 * P = P - K * S * K^T
 *
 * @param covariances 共分散行列 [num_targets * STATE_DIM * STATE_DIM]（入出力）
 * @param kalman_gain カルマンゲイン [num_targets * STATE_DIM * MEAS_DIM]
 * @param innovation_cov イノベーション共分散 [num_targets * MEAS_DIM * MEAS_DIM]
 * @param num_targets 目標数
 */
__global__ void updateCovariance(
    float* covariances,
    const float* kalman_gain,
    const float* innovation_cov,
    int num_targets
);

/**
 * @brief 融合カーネル: 予測ステップの全計算を1カーネルで実行（最適化版）
 *
 * 各ブロックが1つの目標を処理し、スレッド間で並列計算を実行します。
 * メモリアクセスを削減し、レイテンシを最小化します。
 *
 * @param states 状態ベクトル [num_targets * STATE_DIM]（入出力）
 * @param covariances 共分散行列 [num_targets * STATE_DIM * STATE_DIM]（入出力）
 * @param process_cov プロセスノイズ共分散 [STATE_DIM * STATE_DIM]
 * @param weights_mean 平均重み [SIGMA_POINTS]
 * @param weights_cov 共分散重み [SIGMA_POINTS]
 * @param num_targets 目標数
 * @param dt 時間差分
 * @param lambda UKFパラメータ
 */
__global__ void fusedPredict(
    float* states,
    float* covariances,
    const float* process_cov,
    const float* weights_mean,
    const float* weights_cov,
    int num_targets,
    float dt,
    float lambda
);

} // namespace cuda
} // namespace fasttracker

#endif // FASTTRACKER_UKF_KERNELS_CUH
