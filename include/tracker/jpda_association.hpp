#ifndef FASTTRACKER_JPDA_ASSOCIATION_HPP
#define FASTTRACKER_JPDA_ASSOCIATION_HPP

#include "utils/types.hpp"
#include "tracker/data_association.cuh"  // AssociationResult
#include <vector>

namespace fasttracker {

// JPDA結果: 各トラックのβ重みと合成観測
struct JPDATrackUpdate {
    int track_index;              // tracks配列内のインデックス
    bool has_gated_meas;          // ゲート内に観測があったか
    float beta_0;                 // 無観測(miss)の事後確率
    MeasVector combined_meas;     // β重み付き合成観測 (UKFに渡す)
    int best_meas_index;          // 最大β観測のインデックス (-1=なし)
};

struct JPDAResult {
    AssociationResult base_result;  // GNN互換: unassigned_tracks, unassigned_measurements
    std::vector<JPDATrackUpdate> track_updates;
};

/**
 * @brief JPDA (Joint Probabilistic Data Association)
 *
 * 各トラックがゲート内の全観測からβ重み付き合成観測を受ける。
 * GNNの1-to-1制約を排除し、近接目標クラスタでの飢餓問題を解決する。
 *
 * 状態更新はGPU UKFで実行（EKF線形化を回避し、UKFのシグマポイント精度を維持）。
 * クロストラック排他性: β_ij = Pd * L_ij / (λ_c + Pd * Σ_k L_kj)
 */
class JPDAAssociation {
public:
    JPDAAssociation(const AssociationParams& params,
                    const MeasurementNoise& meas_noise);

    /**
     * @brief JPDAアソシエーション（β重み計算 + 合成観測生成）
     *
     * 状態更新は行わない。呼び出し元がUKFで合成観測を使って更新する。
     *
     * @param tracks トラックリスト（予測済み）
     * @param measurements 観測リスト
     * @param sensor_x,sensor_y,sensor_z センサー位置
     * @return JPDAResult: β重み + 合成観測 + 未割当情報
     */
    JPDAResult associate(const std::vector<Track>& tracks,
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
};

} // namespace fasttracker

#endif // FASTTRACKER_JPDA_ASSOCIATION_HPP
