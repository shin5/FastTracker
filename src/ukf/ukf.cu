#include "ukf/ukf.cuh"
#include "ukf/ukf_kernels.cuh"
#include <cmath>
#include <iostream>

namespace fasttracker {

UKF::UKF(int max_targets,
         const UKFParams& params,
         const ProcessNoise& process_noise,
         const MeasurementNoise& meas_noise)
    : max_targets_(max_targets),
      params_(params),
      process_noise_(process_noise),
      meas_noise_(meas_noise),
      // デバイスメモリ割り当て
      d_states_(max_targets * STATE_DIM),
      d_covariances_(max_targets * STATE_DIM * STATE_DIM),
      d_sigma_points_(max_targets * SIGMA_POINTS * STATE_DIM),
      d_weights_mean_(SIGMA_POINTS),
      d_weights_cov_(SIGMA_POINTS),
      d_process_cov_(STATE_DIM * STATE_DIM),
      d_meas_cov_(MEAS_DIM * MEAS_DIM),
      // 作業用メモリ
      d_pred_sigma_points_(max_targets * SIGMA_POINTS * STATE_DIM),
      d_pred_measurements_(max_targets * SIGMA_POINTS * MEAS_DIM),
      d_pred_mean_(max_targets * STATE_DIM),
      d_pred_cov_(max_targets * STATE_DIM * STATE_DIM),
      d_meas_mean_(max_targets * MEAS_DIM),
      d_innovation_cov_(max_targets * MEAS_DIM * MEAS_DIM),
      d_cross_cov_(max_targets * STATE_DIM * MEAS_DIM),
      d_kalman_gain_(max_targets * STATE_DIM * MEAS_DIM)
{
    // 重みの初期化
    initializeWeights();

    // プロセスノイズ共分散の初期化
    initializeProcessCov();

    // 観測ノイズ共分散の初期化
    initializeMeasCov();

    std::cout << "UKF initialized for " << max_targets << " targets" << std::endl;
    std::cout << "  Alpha: " << params_.alpha << std::endl;
    std::cout << "  Beta: " << params_.beta << std::endl;
    std::cout << "  Lambda: " << params_.lambda << std::endl;
}

UKF::~UKF() {
    // DeviceMemoryのデストラクタが自動的にメモリを解放
}

void UKF::initializeWeights() {
    // 重みの計算（ホスト側）
    std::vector<float> w_mean(SIGMA_POINTS);
    std::vector<float> w_cov(SIGMA_POINTS);

    float lambda = params_.lambda;
    int n = STATE_DIM;

    // W_0
    w_mean[0] = lambda / (n + lambda);
    w_cov[0] = w_mean[0] + (1.0f - params_.alpha * params_.alpha + params_.beta);

    // W_i, i = 1, ..., 2n
    float w_i = 1.0f / (2.0f * (n + lambda));
    for (int i = 1; i < SIGMA_POINTS; i++) {
        w_mean[i] = w_i;
        w_cov[i] = w_i;
    }

    // デバイスへコピー
    d_weights_mean_.copyFrom(w_mean.data(), SIGMA_POINTS);
    d_weights_cov_.copyFrom(w_cov.data(), SIGMA_POINTS);
}

void UKF::initializeProcessCov() {
    // プロセスノイズ共分散行列 Q (9×9)
    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> Q;
    Q.setZero();

    float pos_var = process_noise_.position_noise * process_noise_.position_noise;
    float vel_var = process_noise_.velocity_noise * process_noise_.velocity_noise;
    float acc_var = process_noise_.accel_noise * process_noise_.accel_noise;

    // 位置ノイズ (x, y, z)
    Q(0, 0) = pos_var;
    Q(1, 1) = pos_var;
    Q(2, 2) = pos_var;

    // 速度ノイズ (vx, vy, vz)
    Q(3, 3) = vel_var;
    Q(4, 4) = vel_var;
    Q(5, 5) = vel_var;

    // 加速度ノイズ (ax, ay, az)
    Q(6, 6) = acc_var;
    Q(7, 7) = acc_var;
    Q(8, 8) = acc_var;

    // デバイスへコピー
    d_process_cov_.copyFrom(Q.data(), STATE_DIM * STATE_DIM);
}

void UKF::initializeMeasCov() {
    // 観測ノイズ共分散行列 R
    Eigen::Matrix<float, MEAS_DIM, MEAS_DIM> R;
    R.setZero();

    R(0, 0) = meas_noise_.range_noise * meas_noise_.range_noise;
    R(1, 1) = meas_noise_.azimuth_noise * meas_noise_.azimuth_noise;
    R(2, 2) = meas_noise_.elevation_noise * meas_noise_.elevation_noise;
    R(3, 3) = meas_noise_.doppler_noise * meas_noise_.doppler_noise;

    // デバイスへコピー
    d_meas_cov_.copyFrom(R.data(), MEAS_DIM * MEAS_DIM);
}

void UKF::predict(float* states, float* covariances, int num_targets, float dt,
                  int model_id) {
    if (num_targets > max_targets_) {
        throw std::runtime_error("Number of targets exceeds maximum");
    }

    // カーネル起動設定
    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;

    // 1. シグマポイント生成
    cuda::generateSigmaPoints<<<grid_size, block_size, 0, stream_predict_.get()>>>(
        states, covariances, d_sigma_points_.get(), num_targets, params_.lambda
    );

    // 2. シグマポイント予測（状態遷移）
    int sigma_block_size = 256;
    int sigma_grid_size = (num_targets * SIGMA_POINTS + sigma_block_size - 1) / sigma_block_size;
    cuda::predictSigmaPoints<<<sigma_grid_size, sigma_block_size, 0, stream_predict_.get()>>>(
        d_sigma_points_.get(), d_pred_sigma_points_.get(), num_targets, dt, model_id
    );

    // 3. 予測平均計算
    cuda::computeWeightedMean<<<grid_size, block_size, 0, stream_predict_.get()>>>(
        d_pred_sigma_points_.get(), d_weights_mean_.get(),
        d_pred_mean_.get(), num_targets, STATE_DIM
    );

    // 4. 予測共分散計算
    cuda::computeCovariance<<<grid_size, block_size, 0, stream_predict_.get()>>>(
        d_pred_sigma_points_.get(), d_pred_mean_.get(), d_weights_cov_.get(),
        d_pred_cov_.get(), num_targets, STATE_DIM
    );

    // 5. プロセスノイズ加算（dtスケーリング）
    cuda::addNoiseCov<<<grid_size, block_size, 0, stream_predict_.get()>>>(
        d_pred_cov_.get(), d_process_cov_.get(), num_targets, STATE_DIM, dt
    );

    // 予測結果をstatesとcovariancesにコピー
    CUDA_CHECK(cudaMemcpyAsync(states, d_pred_mean_.get(),
                               num_targets * STATE_DIM * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_predict_.get()));

    CUDA_CHECK(cudaMemcpyAsync(covariances, d_pred_cov_.get(),
                               num_targets * STATE_DIM * STATE_DIM * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_predict_.get()));

    // ストリーム同期
    stream_predict_.synchronize();
}

void UKF::update(float* states, float* covariances,
                 const float* measurements, int num_targets,
                 float sensor_x, float sensor_y, float sensor_z) {
    if (num_targets > max_targets_) {
        throw std::runtime_error("Number of targets exceeds maximum");
    }

    // カーネル起動設定
    int block_size = 256;
    int grid_size = (num_targets + block_size - 1) / block_size;

    // 1. シグマポイント生成（現在の状態から）
    cuda::generateSigmaPoints<<<grid_size, block_size, 0, stream_update_.get()>>>(
        states, covariances, d_sigma_points_.get(), num_targets, params_.lambda
    );

    // 2. 観測モデル適用
    int sigma_block_size = 256;
    int sigma_grid_size = (num_targets * SIGMA_POINTS + sigma_block_size - 1) / sigma_block_size;
    cuda::measurementModel<<<sigma_grid_size, sigma_block_size, 0, stream_update_.get()>>>(
        d_sigma_points_.get(), d_pred_measurements_.get(), num_targets,
        sensor_x, sensor_y, sensor_z
    );

    // 3. 予測観測の平均計算
    cuda::computeWeightedMean<<<grid_size, block_size, 0, stream_update_.get()>>>(
        d_pred_measurements_.get(), d_weights_mean_.get(),
        d_meas_mean_.get(), num_targets, MEAS_DIM
    );

    // 4. イノベーション共分散計算
    cuda::computeCovariance<<<grid_size, block_size, 0, stream_update_.get()>>>(
        d_pred_measurements_.get(), d_meas_mean_.get(), d_weights_cov_.get(),
        d_innovation_cov_.get(), num_targets, MEAS_DIM
    );

    // 5. 観測ノイズ加算
    cuda::addNoiseCov<<<grid_size, block_size, 0, stream_update_.get()>>>(
        d_innovation_cov_.get(), d_meas_cov_.get(), num_targets, MEAS_DIM
    );

    // 6. クロス共分散計算
    cuda::computeCrossCov<<<grid_size, block_size, 0, stream_update_.get()>>>(
        d_sigma_points_.get(), d_pred_measurements_.get(),
        states, d_meas_mean_.get(), d_weights_cov_.get(),
        d_cross_cov_.get(), num_targets
    );

    // 7. カルマンゲイン計算
    cuda::computeKalmanGain<<<grid_size, block_size, 0, stream_update_.get()>>>(
        d_cross_cov_.get(), d_innovation_cov_.get(),
        d_kalman_gain_.get(), num_targets
    );

    // 8. 状態更新
    cuda::updateState<<<grid_size, block_size, 0, stream_update_.get()>>>(
        states, d_kalman_gain_.get(), measurements,
        d_meas_mean_.get(), num_targets
    );

    // 9. 共分散更新
    cuda::updateCovariance<<<grid_size, block_size, 0, stream_update_.get()>>>(
        covariances, d_kalman_gain_.get(), d_innovation_cov_.get(), num_targets
    );

    // ストリーム同期
    stream_update_.synchronize();
}

void UKF::predictAndUpdate(float* states, float* covariances,
                           const float* measurements, int num_targets, float dt) {
    // 最適化版: 予測と更新を連続実行
    // 将来的にfusedカーネルを使用することも可能

    predict(states, covariances, num_targets, dt);
    update(states, covariances, measurements, num_targets);
}

void UKF::copyToDevice(const std::vector<StateVector>& host_states,
                       const std::vector<StateCov>& host_covs) {
    if (host_states.size() != host_covs.size()) {
        throw std::runtime_error("States and covariances size mismatch");
    }

    int num_targets = static_cast<int>(host_states.size());
    if (num_targets > max_targets_) {
        throw std::runtime_error("Number of targets exceeds maximum");
    }

    // Eigenの行列をフラット配列に変換
    std::vector<float> flat_states(num_targets * STATE_DIM);
    std::vector<float> flat_covs(num_targets * STATE_DIM * STATE_DIM);

    for (int i = 0; i < num_targets; i++) {
        std::memcpy(&flat_states[i * STATE_DIM],
                    host_states[i].data(), STATE_DIM * sizeof(float));
        std::memcpy(&flat_covs[i * STATE_DIM * STATE_DIM],
                    host_covs[i].data(), STATE_DIM * STATE_DIM * sizeof(float));
    }

    // デバイスへコピー
    d_states_.copyFrom(flat_states.data(), num_targets * STATE_DIM);
    d_covariances_.copyFrom(flat_covs.data(), num_targets * STATE_DIM * STATE_DIM);
}

void UKF::copyToHost(std::vector<StateVector>& host_states,
                     std::vector<StateCov>& host_covs, int num_targets) {
    if (num_targets > max_targets_) {
        throw std::runtime_error("Number of targets exceeds maximum");
    }

    // フラット配列を用意
    std::vector<float> flat_states(num_targets * STATE_DIM);
    std::vector<float> flat_covs(num_targets * STATE_DIM * STATE_DIM);

    // デバイスからコピー
    d_states_.copyTo(flat_states.data(), num_targets * STATE_DIM);
    d_covariances_.copyTo(flat_covs.data(), num_targets * STATE_DIM * STATE_DIM);

    // Eigen行列に変換
    host_states.resize(num_targets);
    host_covs.resize(num_targets);

    for (int i = 0; i < num_targets; i++) {
        std::memcpy(host_states[i].data(),
                    &flat_states[i * STATE_DIM], STATE_DIM * sizeof(float));
        std::memcpy(host_covs[i].data(),
                    &flat_covs[i * STATE_DIM * STATE_DIM],
                    STATE_DIM * STATE_DIM * sizeof(float));
    }
}

} // namespace fasttracker
