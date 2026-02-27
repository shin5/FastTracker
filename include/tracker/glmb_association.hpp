#ifndef FASTTRACKER_GLMB_ASSOCIATION_HPP
#define FASTTRACKER_GLMB_ASSOCIATION_HPP

#include "utils/types.hpp"
#include "tracker/data_association.cuh"  // AssociationResult
#include <vector>
#include <array>
#include <set>
#include <unordered_map>
#include <queue>
#include <random>

namespace fasttracker {

// GLMB: 各トラックの割当情報
struct GLMBTrackUpdate {
    int track_index;              // tracks配列内のインデックス
    bool has_gated_meas;          // ゲート内に観測があったか
    float existence_prob;         // 更新後のベルヌーイ存在確率
    float miss_prob;              // 未検出マージナル確率
    int best_meas_index;          // ベスト仮説の観測インデックス (-1=miss)
    float best_meas_prob;         // ベスト観測のマージナル確率
    float assignment_confidence;  // 仮説間の割当一致率 (0.0-1.0)
};

// GLMBグローバル仮説: フレーム間で累積するラベル付き仮説
struct GLMBGlobalHypothesis {
    int id;
    float accumulated_score;      // 累積対数尤度
    std::set<int> active_labels;  // アクティブなラベル集合（カーディナリティ推定用）
    // track_id → meas_index (-1=miss)
    std::unordered_map<int, int> track_to_meas;
    // track_id → 観測位置 [range, az, el, doppler]
    std::unordered_map<int, std::array<float, 4>> track_to_meas_pos;
};

// GLMB結果: GNN互換base + 各トラック情報 + カーディナリティ推定
struct GLMBResult {
    AssociationResult base_result;
    std::vector<GLMBTrackUpdate> track_updates;
    float estimated_num_targets;  // カーディナリティ推定値 E[N]
    std::vector<float> cardinality_distribution; // P(N=n) for n=0..num_tracks
};

/**
 * @brief GLMB (Generalized Labeled Multi-Bernoulli) アソシエーション
 *
 * δ-GLMB truncation アルゴリズム (Vo & Vo 2013):
 * - 各フレームでMurty's K-best割当を生成
 * - 前フレームのM個仮説 × K個新規割当 = M×K候補
 * - 累積スコアで上位M個を保持（プルーニング）
 * - 各仮説にラベル集合 (active_labels) を付与
 * - マージナル存在確率 = Σ(含む仮説の重み)
 * - カーディナリティ分布 p(N) = Σ(|active_labels|==N の仮説の重み)
 *
 * PMBM/MHTとの違い:
 *   - ラベル付きRFS: track.idを永続ラベルとして管理
 *   - 明示的カーディナリティ推定
 *   - フレーム間ラベル一貫性
 */
class GLMBAssociation {
public:
    GLMBAssociation(const AssociationParams& params,
                    const MeasurementNoise& meas_noise);

    GLMBResult associate(const std::vector<Track>& tracks,
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

    // --- GLMB固有: フレーム間永続状態 ---
    std::vector<GLMBGlobalHypothesis> hypotheses_;
    int next_hyp_id_ = 0;
    int frame_count_ = 0;

    // --- スタンドアロン実装（MHT/PMBMと同一パターン） ---

    MeasVector predictMeasurement(const StateVector& state,
                                   float sensor_x, float sensor_y, float sensor_z) const;

    float computeNormalizedDist(const MeasVector& pred_meas,
                                 const Measurement& meas) const;

    std::vector<int> hungarianSolve(const std::vector<float>& cost_matrix, int n, int m);

    struct Assignment {
        std::vector<int> track_to_meas;
        float total_cost;
    };

    struct MurtyNode {
        std::vector<float> cost_matrix;
        int n, m;
        float base_cost;
        std::vector<std::pair<int,int>> fixed_assignments;

        bool operator>(const MurtyNode& other) const {
            return base_cost > other.base_cost;
        }
    };

    std::vector<Assignment> murtyKBest(
        const std::vector<float>& cost_matrix,
        int num_tracks, int num_cols, int K);

    std::vector<Assignment> gibbsSample(
        const std::vector<float>& cost_matrix,
        int num_tracks, int num_cols, int K);

    std::vector<Assignment> generateAssignments(
        const std::vector<float>& cost_matrix,
        int num_tracks, int num_cols, int K);

    // GLMB固有: カーディナリティ分布計算
    std::vector<float> computeCardinalityDistribution(
        const std::vector<GLMBGlobalHypothesis>& hyps,
        const std::vector<float>& weights,
        int max_n) const;

    // Gibbs sampler用の乱数生成器
    std::mt19937 rng_{42};
};

} // namespace fasttracker

#endif // FASTTRACKER_GLMB_ASSOCIATION_HPP
