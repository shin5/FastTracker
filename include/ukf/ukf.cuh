#ifndef FASTTRACKER_UKF_CUH
#define FASTTRACKER_UKF_CUH

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <vector>
#include "utils/types.hpp"
#include "utils/cuda_utils.cuh"

namespace fasttracker {

/**
 * @brief GPU加速されたUnscented Kalman Filter
 *
 * 複数の目標に対してバッチ処理でUKFを適用します。
 * CUDA並列化により、大規模な多目標追尾を高速化します。
 */
class UKF {
public:
    /**
     * @brief コンストラクタ
     * @param max_targets 最大目標数
     * @param params UKFパラメータ
     * @param process_noise プロセスノイズパラメータ
     * @param meas_noise 観測ノイズパラメータ
     */
    UKF(int max_targets,
        const UKFParams& params = UKFParams(),
        const ProcessNoise& process_noise = ProcessNoise(),
        const MeasurementNoise& meas_noise = MeasurementNoise());

    ~UKF();

    // コピー禁止
    UKF(const UKF&) = delete;
    UKF& operator=(const UKF&) = delete;

    /**
     * @brief 予測ステップ（バッチ処理）
     * @param states 状態ベクトル配列 [num_targets * STATE_DIM]
     * @param covariances 共分散行列配列 [num_targets * STATE_DIM * STATE_DIM]
     * @param num_targets 目標数
     * @param dt 時間差分 [s]
     */
    void predict(float* states, float* covariances, int num_targets, float dt);

    /**
     * @brief 更新ステップ（バッチ処理）
     * @param states 状態ベクトル配列 [num_targets * STATE_DIM]
     * @param covariances 共分散行列配列 [num_targets * STATE_DIM * STATE_DIM]
     * @param measurements 観測ベクトル配列 [num_targets * MEAS_DIM]
     * @param num_targets 目標数
     */
    void update(float* states, float* covariances,
                const float* measurements, int num_targets,
                float sensor_x = 0.0f, float sensor_y = 0.0f);

    /**
     * @brief 予測+更新を一度に実行（最適化版）
     * @param states 状態ベクトル配列
     * @param covariances 共分散行列配列
     * @param measurements 観測ベクトル配列
     * @param num_targets 目標数
     * @param dt 時間差分 [s]
     */
    void predictAndUpdate(float* states, float* covariances,
                          const float* measurements, int num_targets, float dt);

    /**
     * @brief ホストメモリからGPUメモリへデータ転送
     * @param host_states ホスト側状態配列
     * @param host_covs ホスト側共分散配列
     * @param num_targets 目標数
     */
    void copyToDevice(const std::vector<StateVector>& host_states,
                      const std::vector<StateCov>& host_covs);

    /**
     * @brief GPUメモリからホストメモリへデータ転送
     * @param host_states ホスト側状態配列（出力）
     * @param host_covs ホスト側共分散配列（出力）
     * @param num_targets 目標数
     */
    void copyToHost(std::vector<StateVector>& host_states,
                    std::vector<StateCov>& host_covs, int num_targets);

    /**
     * @brief UKFパラメータを取得
     */
    const UKFParams& getParams() const { return params_; }

    /**
     * @brief プロセスノイズパラメータを取得
     */
    const ProcessNoise& getProcessNoise() const { return process_noise_; }

    /**
     * @brief 観測ノイズパラメータを取得
     */
    const MeasurementNoise& getMeasNoise() const { return meas_noise_; }

    /**
     * @brief 最大目標数を取得
     */
    int getMaxTargets() const { return max_targets_; }

    /**
     * @brief デバイスメモリポインタを取得（デバッグ用）
     */
    float* getDeviceStates() { return d_states_.get(); }
    float* getDeviceCovariances() { return d_covariances_.get(); }

private:
    // パラメータ
    int max_targets_;
    UKFParams params_;
    ProcessNoise process_noise_;
    MeasurementNoise meas_noise_;

    // デバイスメモリ
    cuda::DeviceMemory<float> d_states_;          // [max_targets * STATE_DIM]
    cuda::DeviceMemory<float> d_covariances_;     // [max_targets * STATE_DIM * STATE_DIM]
    cuda::DeviceMemory<float> d_sigma_points_;    // [max_targets * SIGMA_POINTS * STATE_DIM]
    cuda::DeviceMemory<float> d_weights_mean_;    // [SIGMA_POINTS]
    cuda::DeviceMemory<float> d_weights_cov_;     // [SIGMA_POINTS]
    cuda::DeviceMemory<float> d_process_cov_;     // [STATE_DIM * STATE_DIM]
    cuda::DeviceMemory<float> d_meas_cov_;        // [MEAS_DIM * MEAS_DIM]

    // 作業用メモリ
    cuda::DeviceMemory<float> d_pred_sigma_points_;  // [max_targets * SIGMA_POINTS * STATE_DIM]
    cuda::DeviceMemory<float> d_pred_measurements_;  // [max_targets * SIGMA_POINTS * MEAS_DIM]
    cuda::DeviceMemory<float> d_pred_mean_;          // [max_targets * STATE_DIM]
    cuda::DeviceMemory<float> d_pred_cov_;           // [max_targets * STATE_DIM * STATE_DIM]
    cuda::DeviceMemory<float> d_meas_mean_;          // [max_targets * MEAS_DIM]
    cuda::DeviceMemory<float> d_innovation_cov_;     // [max_targets * MEAS_DIM * MEAS_DIM]
    cuda::DeviceMemory<float> d_cross_cov_;          // [max_targets * STATE_DIM * MEAS_DIM]
    cuda::DeviceMemory<float> d_kalman_gain_;        // [max_targets * STATE_DIM * MEAS_DIM]

    // CUDAストリーム（パイプライン処理用）
    cuda::CudaStream stream_predict_;
    cuda::CudaStream stream_update_;

    /**
     * @brief 重みの初期化
     */
    void initializeWeights();

    /**
     * @brief プロセスノイズ共分散行列の初期化
     */
    void initializeProcessCov();

    /**
     * @brief 観測ノイズ共分散行列の初期化
     */
    void initializeMeasCov();
};

} // namespace fasttracker

#endif // FASTTRACKER_UKF_CUH
