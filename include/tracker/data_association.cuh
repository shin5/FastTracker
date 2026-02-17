#ifndef FASTTRACKER_DATA_ASSOCIATION_CUH
#define FASTTRACKER_DATA_ASSOCIATION_CUH

#include <cuda_runtime.h>
#include <vector>
#include "utils/types.hpp"
#include "utils/cuda_utils.cuh"

namespace fasttracker {

/**
 * @brief データアソシエーション結果
 */
struct AssociationResult {
    std::vector<int> track_to_meas;  // track_id -> measurement_index (-1 = 未割り当て)
    std::vector<int> meas_to_track;  // measurement_index -> track_id (-1 = 未割り当て)
    std::vector<int> unassigned_tracks;     // 未割り当てトラック
    std::vector<int> unassigned_measurements;  // 未割り当て観測（新規トラック候補）
};

/**
 * @brief GPU加速データアソシエーション
 *
 * Global Nearest Neighbor (GNN) アルゴリズムを使用します。
 * Mahalanobis距離計算とHungarian法をGPUで高速化します。
 */
class DataAssociation {
public:
    /**
     * @brief コンストラクタ
     * @param max_tracks 最大トラック数
     * @param max_measurements 最大観測数
     * @param params アソシエーションパラメータ
     */
    DataAssociation(int max_tracks, int max_measurements,
                    const AssociationParams& params = AssociationParams());

    ~DataAssociation();

    // コピー禁止
    DataAssociation(const DataAssociation&) = delete;
    DataAssociation& operator=(const DataAssociation&) = delete;

    /**
     * @brief データアソシエーションを実行
     * @param tracks トラックリスト
     * @param measurements 観測リスト
     * @return アソシエーション結果
     */
    AssociationResult associate(const std::vector<Track>& tracks,
                                 const std::vector<Measurement>& measurements);

    /**
     * @brief センサー位置を設定
     */
    void setSensorPosition(float x, float y, float z = 0.0f) { sensor_x_ = x; sensor_y_ = y; sensor_z_ = z; }

    /**
     * @brief 観測ノイズを設定
     */
    void setMeasurementNoise(const MeasurementNoise& noise) { meas_noise_ = noise; }

    /**
     * @brief Mahalanobis距離を計算（CPU版、デバッグ用）
     * @param track トラック
     * @param meas 観測
     * @return Mahalanobis距離
     */
    static float computeMahalanobisDistance(const Track& track,
                                            const Measurement& meas);

private:
    int max_tracks_;
    int max_measurements_;
    AssociationParams params_;
    MeasurementNoise meas_noise_;
    float sensor_x_ = 0.0f;
    float sensor_y_ = 0.0f;
    float sensor_z_ = 0.0f;

    // デバイスメモリ
    cuda::DeviceMemory<float> d_track_states_;      // [max_tracks * STATE_DIM]
    cuda::DeviceMemory<float> d_track_covs_;        // [max_tracks * STATE_DIM * STATE_DIM]
    cuda::DeviceMemory<float> d_measurements_;      // [max_measurements * MEAS_DIM]
    cuda::DeviceMemory<float> d_cost_matrix_;       // [max_tracks * max_measurements]
    cuda::DeviceMemory<int> d_assignments_;         // [max_tracks]

    // 作業用メモリ
    cuda::DeviceMemory<float> d_pred_meas_;         // [max_tracks * MEAS_DIM]
    cuda::DeviceMemory<float> d_innovation_covs_;   // [max_tracks * MEAS_DIM * MEAS_DIM]

    /**
     * @brief コスト行列を計算（GPU）
     */
    void computeCostMatrix(int num_tracks, int num_measurements);

    /**
     * @brief Hungarian法で最適割り当てを計算
     * @param num_tracks トラック数
     * @param num_measurements 観測数
     * @return 割り当て結果 [num_tracks]（-1 = 未割り当て）
     */
    std::vector<int> hungarianAssignment(int num_tracks, int num_measurements);

    /**
     * @brief Hungarian法（CPU版）
     * @param cost_matrix コスト行列 [n x m]
     * @param n 行数
     * @param m 列数
     * @return 割り当て結果
     */
    static std::vector<int> hungarianAlgorithm(const std::vector<float>& cost_matrix,
                                               int n, int m);
};

// CUDAカーネル宣言
namespace cuda {

/**
 * @brief 予測観測を計算するカーネル
 */
__global__ void predictMeasurements(
    const float* track_states,
    float* pred_measurements,
    int num_tracks,
    float sensor_x = 0.0f,
    float sensor_y = 0.0f,
    float sensor_z = 0.0f
);

/**
 * @brief イノベーション共分散を計算するカーネル
 */
__global__ void computeInnovationCovs(
    const float* track_covs,
    const float* meas_noise_cov,
    float* innovation_covs,
    int num_tracks
);

/**
 * @brief Mahalanobis距離コスト行列を計算するカーネル
 */
__global__ void computeMahalanobisCostMatrix(
    const float* pred_measurements,
    const float* measurements,
    const float* meas_noise_stds,
    float* cost_matrix,
    int num_tracks,
    int num_measurements,
    float gate_threshold
);

} // namespace cuda
} // namespace fasttracker

#endif // FASTTRACKER_DATA_ASSOCIATION_CUH
