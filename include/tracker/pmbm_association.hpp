#ifndef FASTTRACKER_PMBM_ASSOCIATION_HPP
#define FASTTRACKER_PMBM_ASSOCIATION_HPP

#include "utils/types.hpp"
#include "tracker/data_association.cuh"  // AssociationResult
#include <vector>
#include <queue>

namespace fasttracker {

// PMBM結果: 各トラックの存在確率とベスト割当
struct PMBMTrackUpdate {
    int track_index;              // tracks配列内のインデックス
    bool has_gated_meas;          // ゲート内に観測があったか
    float existence_prob;         // 更新後のベルヌーイ存在確率
    float miss_prob;              // 未検出マージナル確率
    int best_meas_index;          // ベスト仮説の観測インデックス (-1=miss)
    float best_meas_prob;         // ベスト観測のマージナル確率
};

struct PMBMResult {
    AssociationResult base_result;  // GNN互換: unassigned_tracks, unassigned_measurements
    std::vector<PMBMTrackUpdate> track_updates;
};

/**
 * @brief PMBM (Poisson Multi-Bernoulli Mixture) アソシエーション
 *
 * Track-oriented PMB近似:
 * - Murty's K-best割当でトップK仮説を列挙
 * - マージナル割当確率を仮説集合から計算
 * - ベルヌーイ存在確率で航跡ライフサイクルを管理
 *
 * 状態更新は呼び出し元がGPU UKFで実行（GNN/JPDAと同じパス）。
 */
class PMBMAssociation {
public:
    PMBMAssociation(const AssociationParams& params,
                    const MeasurementNoise& meas_noise);

    /**
     * @brief PMBMアソシエーション
     *
     * @param tracks トラックリスト（予測済み）
     * @param measurements 観測リスト
     * @param sensor_x,sensor_y,sensor_z センサー位置
     * @return PMBMResult: ベスト仮説割当 + 存在確率 + 未割当情報
     */
    PMBMResult associate(const std::vector<Track>& tracks,
                         const std::vector<Measurement>& measurements,
                         float sensor_x, float sensor_y, float sensor_z);

    void setSensorPosition(float x, float y, float z) {
        sensor_x_ = x; sensor_y_ = y; sensor_z_ = z;
    }

private:
    AssociationParams params_;
    MeasurementNoise meas_noise_;
    float sensor_x_ = 0.0f;
    float sensor_y_ = 0.0f;
    float sensor_z_ = 0.0f;

    // 予測観測値を計算（状態 → [range, az, el, doppler]）
    MeasVector predictMeasurement(const StateVector& state,
                                   float sensor_x, float sensor_y, float sensor_z) const;

    // 正規化距離（ゲーティング用）
    float computeNormalizedDist(const MeasVector& pred_meas,
                                 const Measurement& meas) const;

    // スタンドアロンHungarian法（Munkresアルゴリズム）
    std::vector<int> hungarianSolve(const std::vector<float>& cost_matrix, int n, int m);

    // Murty's K-best割当
    struct Assignment {
        std::vector<int> track_to_meas;  // track_i → 列index (-1=none for extended cols)
        float total_cost;
    };

    // Murty分割ノード
    struct MurtyNode {
        std::vector<float> cost_matrix;  // 制約付きコスト行列
        int n, m;                        // 行列サイズ
        float base_cost;                 // 固定済み部分のコスト
        std::vector<std::pair<int,int>> fixed_assignments;  // 固定済み割当

        bool operator>(const MurtyNode& other) const {
            return base_cost > other.base_cost;  // 最小ヒープ用
        }
    };

    std::vector<Assignment> murtyKBest(
        const std::vector<float>& cost_matrix,
        int num_tracks, int num_cols, int K);
};

} // namespace fasttracker

#endif // FASTTRACKER_PMBM_ASSOCIATION_HPP
