#include "tracker/data_association.cuh"
#include "utils/matrix.cuh"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <utility>

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

    // 予測観測を計算（センサー位置基準）
    grid_size = (num_tracks + block_size - 1) / block_size;
    cuda::predictMeasurements<<<grid_size, block_size>>>(
        d_track_states_.get(), d_pred_meas_.get(), num_tracks,
        sensor_x_, sensor_y_, sensor_z_
    );

    // 観測ノイズ標準偏差をデバイスに渡す（正規化距離計算用）
    float meas_noise_stds[MEAS_DIM] = {
        meas_noise_.range_noise,
        meas_noise_.azimuth_noise,
        meas_noise_.elevation_noise,
        meas_noise_.doppler_noise
    };
    float* d_meas_noise_stds = nullptr;
    cudaMalloc(&d_meas_noise_stds, MEAS_DIM * sizeof(float));
    cudaMemcpy(d_meas_noise_stds, meas_noise_stds, MEAS_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // コスト行列を計算（正規化距離）
    grid_size = (num_tracks * num_measurements + block_size - 1) / block_size;
    cuda::computeMahalanobisCostMatrix<<<grid_size, block_size>>>(
        d_pred_meas_.get(),
        d_measurements_.get(),
        d_meas_noise_stds,
        d_cost_matrix_.get(),
        num_tracks,
        num_measurements,
        params_.gate_threshold
    );

    cudaFree(d_meas_noise_stds);

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
    // Munkres (Hungarian) アルゴリズム — 最適割り当て
    std::vector<int> assignments(n, -1);
    if (n == 0 || m == 0) return assignments;

    const int N = std::max(n, m);

    // パディング済み正方コスト行列を構築
    std::vector<float> C(N * N, 0.0f);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i * N + j] = cost_matrix[i * m + j];

    // スター・プライム・カバー
    std::vector<int> star_col(N, -1);   // star_col[i] = 行iのスター列 (-1=なし)
    std::vector<int> star_row(N, -1);   // star_row[j] = 列jのスター行 (-1=なし)
    std::vector<int> prime_col(N, -1);  // prime_col[i] = 行iのプライム列
    std::vector<bool> row_cover(N, false);
    std::vector<bool> col_cover(N, false);

    // Step 0: 行リダクション
    for (int i = 0; i < N; i++) {
        float row_min = C[i * N];
        for (int j = 1; j < N; j++)
            if (C[i * N + j] < row_min) row_min = C[i * N + j];
        for (int j = 0; j < N; j++)
            C[i * N + j] -= row_min;
    }

    // Step 1: 列リダクション
    for (int j = 0; j < N; j++) {
        float col_min = C[j];
        for (int i = 1; i < N; i++)
            if (C[i * N + j] < col_min) col_min = C[i * N + j];
        for (int i = 0; i < N; i++)
            C[i * N + j] -= col_min;
    }

    // Step 2: 初期スター化
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] == 0.0f && star_col[i] == -1 && star_row[j] == -1) {
                star_col[i] = j;
                star_row[j] = i;
            }
        }
    }

    // メインループ
    for (;;) {
        // Step 3: スター列をカバー
        std::fill(col_cover.begin(), col_cover.end(), false);
        int covered = 0;
        for (int i = 0; i < N; i++) {
            if (star_col[i] >= 0) {
                col_cover[star_col[i]] = true;
                covered++;
            }
        }
        if (covered >= N) break;  // 全行にスターあり → 完了

        // Step 4-6 ループ
        for (;;) {
            // Step 4: 未カバーゼロを探してプライム
            int pr = -1, pc = -1;
            for (;;) {
                pr = -1; pc = -1;
                for (int i = 0; i < N && pr == -1; i++) {
                    if (row_cover[i]) continue;
                    for (int j = 0; j < N; j++) {
                        if (!col_cover[j] && C[i * N + j] == 0.0f) {
                            pr = i; pc = j;
                            break;
                        }
                    }
                }
                if (pr == -1) {
                    // Step 6: 未カバー最小値で調整
                    float min_val = std::numeric_limits<float>::max();
                    for (int i = 0; i < N; i++) {
                        if (row_cover[i]) continue;
                        for (int j = 0; j < N; j++) {
                            if (!col_cover[j] && C[i * N + j] < min_val)
                                min_val = C[i * N + j];
                        }
                    }
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            if (row_cover[i]) C[i * N + j] += min_val;
                            if (!col_cover[j]) C[i * N + j] -= min_val;
                        }
                    }
                    // Step 4 を再試行
                    continue;
                }
                break;
            }

            // プライムを記録
            prime_col[pr] = pc;

            if (star_col[pr] == -1) {
                // Step 5: 増加パス（プライムから始まる交互パス）
                int path_row = pr, path_col = pc;
                for (;;) {
                    int sr = star_row[path_col];
                    // 旧スターを除去
                    if (sr >= 0) {
                        star_col[sr] = -1;
                    }
                    // プライムをスターに昇格
                    star_col[path_row] = path_col;
                    star_row[path_col] = path_row;

                    if (sr < 0) break;
                    // sr の行のプライム列を辿る
                    path_col = prime_col[sr];
                    path_row = sr;
                }

                // プライムとカバーをリセット
                std::fill(prime_col.begin(), prime_col.end(), -1);
                std::fill(row_cover.begin(), row_cover.end(), false);
                std::fill(col_cover.begin(), col_cover.end(), false);
                break;  // Step 3 に戻る
            } else {
                // 同行のスター列をアンカバー、行をカバー
                col_cover[star_col[pr]] = false;
                row_cover[pr] = true;
                // Step 4 を続行（内側ループの次の反復）
            }
        }
    }

    // 結果抽出（ゲートフィルタ付き）
    for (int i = 0; i < n; i++) {
        int j = star_col[i];
        if (j >= 0 && j < m && cost_matrix[i * m + j] < 1e9f) {
            assignments[i] = j;
        }
    }

    return assignments;
}

float DataAssociation::computeMahalanobisDistance(const Track& track,
                                                  const Measurement& meas) {
    // 予測観測（3D）
    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    MeasVector z_pred;
    float x = track.state(0);
    float y = track.state(1);
    float z = track.state(2);
    float vx = track.state(3);
    float vy = track.state(4);
    float vz = track.state(5);

    float range_horiz = std::sqrt(x * x + y * y);
    float range = std::sqrt(x * x + y * y + z * z);
    z_pred(0) = range;
    z_pred(1) = std::atan2(y, x);
    z_pred(2) = (range_horiz > 1e-6f) ? std::atan2(z, range_horiz) : 0.0f;
    z_pred(3) = (range > 1e-6f) ? ((x * vx + y * vy + z * vz) / range) : 0.0f;

    // 実際の観測
    MeasVector z_meas;
    z_meas(0) = meas.range;
    z_meas(1) = meas.azimuth;
    z_meas(2) = meas.elevation;
    z_meas(3) = meas.doppler;

    // イノベーション
    MeasVector innovation = z_meas - z_pred;

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
    int num_tracks,
    float sensor_x,
    float sensor_y,
    float sensor_z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tracks) return;

    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    const float* state = &track_states[tid * STATE_DIM];
    float* pred_meas = &pred_measurements[tid * MEAS_DIM];

    // センサーからの相対座標（3D）
    float dx = state[0] - sensor_x;
    float dy = state[1] - sensor_y;
    float dz = state[2] - sensor_z;
    float vx = state[3];
    float vy = state[4];
    float vz = state[5];

    float range_horiz = sqrtf(dx * dx + dy * dy);
    float range = sqrtf(dx * dx + dy * dy + dz * dz);
    float azimuth = atan2f(dy, dx);
    float elevation = (range_horiz > 1e-6f) ? atan2f(dz, range_horiz) : 0.0f;
    float doppler = (range > 1e-6f) ? ((dx * vx + dy * vy + dz * vz) / range) : 0.0f;

    pred_meas[0] = range;
    pred_meas[1] = azimuth;
    pred_meas[2] = elevation;
    pred_meas[3] = doppler;
}

__global__ void computeMahalanobisCostMatrix(
    const float* pred_measurements,
    const float* measurements,
    const float* meas_noise_stds,
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
    if (innovation[1] > static_cast<float>(M_PI))  innovation[1] -= 2.0f * static_cast<float>(M_PI);
    if (innovation[1] < -static_cast<float>(M_PI)) innovation[1] += 2.0f * static_cast<float>(M_PI);

    // 正規化距離（固定観測ノイズ）
    float distance_sq = 0.0f;
    for (int i = 0; i < MEAS_DIM; i++) {
        float normalized = innovation[i] / meas_noise_stds[i];
        distance_sq += normalized * normalized;
    }

    float distance = sqrtf(distance_sq);

    // ゲーティング
    if (distance > gate_threshold) {
        cost_matrix[tid] = 1e10f;
    } else {
        cost_matrix[tid] = distance;
    }
}

} // namespace cuda
} // namespace fasttracker
