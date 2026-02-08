#include "tracker/data_association.cuh"
#include "utils/matrix.cuh"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

DataAssociation::DataAssociation(int max_tracks, int max_measurements,
                                 const AssociationParams& params)
    : max_tracks_(max_tracks),
      max_measurements_(max_measurements),
      params_(params),
      d_track_states_(max_tracks * STATE_DIM),
      d_track_covs_(max_tracks * STATE_DIM * STATE_DIM),
      d_measurements_(max_measurements * MEAS_DIM),
      d_cost_matrix_(max_tracks * max_measurements),
      d_assignments_(max_tracks),
      d_pred_meas_(max_tracks * MEAS_DIM),
      d_innovation_covs_(max_tracks * MEAS_DIM * MEAS_DIM)
{
}

DataAssociation::~DataAssociation() {
}

AssociationResult DataAssociation::associate(const std::vector<Track>& tracks,
                                             const std::vector<Measurement>& measurements) {
    int num_tracks = static_cast<int>(tracks.size());
    int num_meas = static_cast<int>(measurements.size());

    AssociationResult result;
    result.track_to_meas.resize(num_tracks, -1);
    result.meas_to_track.resize(num_meas, -1);

    if (num_tracks == 0 || num_meas == 0) {
        for (int i = 0; i < num_tracks; i++) {
            result.unassigned_tracks.push_back(i);
        }
        for (int i = 0; i < num_meas; i++) {
            result.unassigned_measurements.push_back(i);
        }
        return result;
    }

    // トラック状態をデバイスにコピー
    std::vector<float> flat_states(num_tracks * STATE_DIM);
    std::vector<float> flat_covs(num_tracks * STATE_DIM * STATE_DIM);
    for (int i = 0; i < num_tracks; i++) {
        std::memcpy(&flat_states[i * STATE_DIM],
                    tracks[i].state.data(), STATE_DIM * sizeof(float));
        std::memcpy(&flat_covs[i * STATE_DIM * STATE_DIM],
                    tracks[i].covariance.data(),
                    STATE_DIM * STATE_DIM * sizeof(float));
    }
    d_track_states_.copyFrom(flat_states.data(), num_tracks * STATE_DIM);
    d_track_covs_.copyFrom(flat_covs.data(), num_tracks * STATE_DIM * STATE_DIM);

    // 観測をデバイスにコピー
    std::vector<float> flat_meas(num_meas * MEAS_DIM);
    for (int i = 0; i < num_meas; i++) {
        flat_meas[i * MEAS_DIM + 0] = measurements[i].range;
        flat_meas[i * MEAS_DIM + 1] = measurements[i].azimuth;
        flat_meas[i * MEAS_DIM + 2] = measurements[i].elevation;
        flat_meas[i * MEAS_DIM + 3] = measurements[i].doppler;
    }
    d_measurements_.copyFrom(flat_meas.data(), num_meas * MEAS_DIM);

    // コスト行列を計算
    computeCostMatrix(num_tracks, num_meas);

    // Hungarian法で最適割り当て
    std::vector<int> assignments = hungarianAssignment(num_tracks, num_meas);

    // 結果を整理
    for (int i = 0; i < num_tracks; i++) {
        int meas_idx = assignments[i];
        if (meas_idx >= 0) {
            result.track_to_meas[i] = meas_idx;
            result.meas_to_track[meas_idx] = i;
        } else {
            result.unassigned_tracks.push_back(i);
        }
    }

    for (int i = 0; i < num_meas; i++) {
        if (result.meas_to_track[i] == -1) {
            result.unassigned_measurements.push_back(i);
        }
    }

    return result;
}

void DataAssociation::computeCostMatrix(int num_tracks, int num_measurements) {
    int block_size = 256;
    int grid_size;

    // 予測観測を計算
    grid_size = (num_tracks + block_size - 1) / block_size;
    cuda::predictMeasurements<<<grid_size, block_size>>>(
        d_track_states_.get(), d_pred_meas_.get(), num_tracks
    );

    // 観測ノイズ共分散行列を設定
    MeasurementNoise meas_noise;  // デフォルト値を使用
    Eigen::Matrix<float, MEAS_DIM, MEAS_DIM> R;
    R.setZero();
    R(0, 0) = meas_noise.range_noise * meas_noise.range_noise;
    R(1, 1) = meas_noise.azimuth_noise * meas_noise.azimuth_noise;
    R(2, 2) = meas_noise.elevation_noise * meas_noise.elevation_noise;
    R(3, 3) = meas_noise.doppler_noise * meas_noise.doppler_noise;

    // コスト行列を計算
    grid_size = (num_tracks * num_measurements + block_size - 1) / block_size;
    cuda::computeMahalanobisCostMatrix<<<grid_size, block_size>>>(
        d_pred_meas_.get(),
        d_measurements_.get(),
        d_innovation_covs_.get(),
        d_cost_matrix_.get(),
        num_tracks,
        num_measurements,
        params_.gate_threshold
    );

    CUDA_CHECK_KERNEL();
}

std::vector<int> DataAssociation::hungarianAssignment(int num_tracks, int num_measurements) {
    // コスト行列をホストにコピー
    std::vector<float> cost_matrix(num_tracks * num_measurements);
    d_cost_matrix_.copyTo(cost_matrix.data(), num_tracks * num_measurements);

    // Hungarian法を実行
    return hungarianAlgorithm(cost_matrix, num_tracks, num_measurements);
}

std::vector<int> DataAssociation::hungarianAlgorithm(
    const std::vector<float>& cost_matrix, int n, int m)
{
    // 簡易版Hungarian法（Munkresアルゴリズム）
    // 完全な実装の代わりに、貪欲法を使用

    std::vector<int> assignments(n, -1);
    std::vector<bool> meas_assigned(m, false);

    // 各トラックについて最小コストの観測を探す
    for (int i = 0; i < n; i++) {
        float min_cost = std::numeric_limits<float>::max();
        int best_meas = -1;

        for (int j = 0; j < m; j++) {
            if (!meas_assigned[j]) {
                float cost = cost_matrix[i * m + j];
                if (cost < min_cost) {
                    min_cost = cost;
                    best_meas = j;
                }
            }
        }

        // 閾値以下なら割り当て
        if (best_meas >= 0 && min_cost < 1e9f) {
            assignments[i] = best_meas;
            meas_assigned[best_meas] = true;
        }
    }

    return assignments;
}

float DataAssociation::computeMahalanobisDistance(const Track& track,
                                                  const Measurement& meas) {
    // 予測観測
    MeasVector z_pred;
    float x = track.state(0);
    float y = track.state(1);
    float vx = track.state(2);
    float vy = track.state(3);

    float range = std::sqrt(x * x + y * y);
    z_pred(0) = range;
    z_pred(1) = std::atan2(y, x);
    z_pred(2) = 0.0f;
    z_pred(3) = (range > 1e-6f) ? ((x * vx + y * vy) / range) : 0.0f;

    // 実際の観測
    MeasVector z;
    z(0) = meas.range;
    z(1) = meas.azimuth;
    z(2) = meas.elevation;
    z(3) = meas.doppler;

    // イノベーション
    MeasVector innovation = z - z_pred;

    // 角度の正規化
    while (innovation(1) > M_PI) innovation(1) -= 2.0f * M_PI;
    while (innovation(1) < -M_PI) innovation(1) += 2.0f * M_PI;

    // イノベーション共分散（簡易版）
    MeasCov S = MeasCov::Identity() * 100.0f;

    // Mahalanobis距離
    float distance = std::sqrt((innovation.transpose() * S.inverse() * innovation)(0));

    return distance;
}

// CUDAカーネル実装

namespace cuda {

__global__ void predictMeasurements(
    const float* track_states,
    float* pred_measurements,
    int num_tracks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tracks) return;

    const float* state = &track_states[tid * STATE_DIM];
    float* pred_meas = &pred_measurements[tid * MEAS_DIM];

    float x = state[0];
    float y = state[1];
    float vx = state[2];
    float vy = state[3];

    float range = sqrtf(x * x + y * y);
    float azimuth = atan2f(y, x);
    float elevation = 0.0f;
    float doppler = (range > 1e-6f) ? ((x * vx + y * vy) / range) : 0.0f;

    pred_meas[0] = range;
    pred_meas[1] = azimuth;
    pred_meas[2] = elevation;
    pred_meas[3] = doppler;
}

__global__ void computeMahalanobisCostMatrix(
    const float* pred_measurements,
    const float* measurements,
    const float* innovation_covs,
    float* cost_matrix,
    int num_tracks,
    int num_measurements,
    float gate_threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tracks * num_measurements;
    if (tid >= total) return;

    int track_idx = tid / num_measurements;
    int meas_idx = tid % num_measurements;

    const float* pred_meas = &pred_measurements[track_idx * MEAS_DIM];
    const float* meas = &measurements[meas_idx * MEAS_DIM];

    // イノベーション
    float innovation[MEAS_DIM];
    for (int i = 0; i < MEAS_DIM; i++) {
        innovation[i] = meas[i] - pred_meas[i];
    }

    // 角度の正規化
    if (innovation[1] > M_PI) innovation[1] -= 2.0f * M_PI;
    if (innovation[1] < -M_PI) innovation[1] += 2.0f * M_PI;

    // 簡易Mahalanobis距離（単位共分散を仮定）
    float distance_sq = 0.0f;
    for (int i = 0; i < MEAS_DIM; i++) {
        distance_sq += innovation[i] * innovation[i];
    }

    float distance = sqrtf(distance_sq);

    // ゲーティング
    if (distance > gate_threshold) {
        cost_matrix[tid] = 1e10f;  // 大きなコスト
    } else {
        cost_matrix[tid] = distance;
    }
}

} // namespace cuda
} // namespace fasttracker
