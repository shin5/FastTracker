#ifndef FASTTRACKER_MHT_ASSOCIATION_HPP
#define FASTTRACKER_MHT_ASSOCIATION_HPP

#include "utils/types.hpp"
#include "tracker/data_association.cuh"  // AssociationResult
#include <vector>
#include <array>
#include <unordered_map>
#include <queue>

namespace fasttracker {

// MHT: 各トラックの割当情報
struct MHTTrackInfo {
    int track_index;              // tracks配列内のインデックス
    int best_meas_index;          // ベスト仮説の観測インデックス (-1=miss)
    float assignment_confidence;  // 仮説間の割当一致率 (0.0-1.0)
    float miss_prob;              // 仮説のスコア加重ミス確率 (0.0-1.0)
    float existence_prob;         // 更新後の存在確率
};

// MHTグローバル仮説: フレーム間で累積スコアを持つ割当
struct MHTGlobalHypothesis {
    int id;
    float accumulated_score;      // 累積対数尤度（高いほど良い）
    // track_id → meas_index (-1=miss)
    // track ID（永続的）をキーに使用、track index（フレーム毎に変動）ではない
    std::unordered_map<int, int> track_to_meas;
    // track_id → 割り当てられた観測の位置 [range, az, el, doppler]
    // フレーム間で観測インデックスは変動するため、位置で一貫性を判定する
    std::unordered_map<int, std::array<float, 4>> track_to_meas_pos;
};

// MHT結果: GNN互換base + 各トラック情報
struct MHTResult {
    AssociationResult base_result;     // GNN互換: track_to_meas, meas_to_track, unassigned
    std::vector<MHTTrackInfo> track_info;
};

/**
 * @brief MHT (Multiple Hypothesis Tracking) アソシエーション
 *
 * スライディングウィンドウ・グローバル仮説MHT:
 * - 各フレームでMurty's K-best割当を生成
 * - 前フレームのM個仮説 × K個新規割当 = M×K候補
 * - 累積スコアで上位M個を保持（プルーニング）
 * - ベスト仮説の現フレーム割当を返す
 *
 * 一貫した割当に高スコアを与え、ワンダリングを抑制する。
 * 状態更新は呼び出し元がGPU UKFで実行（GNN/JPDA/PMBMと同じパス）。
 */
class MHTAssociation {
public:
    MHTAssociation(const AssociationParams& params,
                   const MeasurementNoise& meas_noise);

    /**
     * @brief MHTアソシエーション
     *
     * @param tracks トラックリスト（予測済み）
     * @param measurements 観測リスト
     * @param sensor_x,sensor_y,sensor_z センサー位置
     * @return MHTResult: ベスト仮説割当 + 割当信頼度 + 未割当情報
     */
    MHTResult associate(const std::vector<Track>& tracks,
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

    // --- MHT固有: フレーム間永続状態 ---
    std::vector<MHTGlobalHypothesis> hypotheses_;
    int next_hyp_id_ = 0;
    int frame_count_ = 0;

    // --- PMBMからコピー（スタンドアロン実装） ---

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

#endif // FASTTRACKER_MHT_ASSOCIATION_HPP
