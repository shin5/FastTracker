#include "tracker/jpda_association.hpp"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

JPDAAssociation::JPDAAssociation(const AssociationParams& params,
                                   const MeasurementNoise& meas_noise)
    : params_(params), meas_noise_(meas_noise)
{
}

MeasVector JPDAAssociation::predictMeasurement(const StateVector& state,
                                                 float sensor_x, float sensor_y, float sensor_z) const
{
    // GPU版 predictMeasurements カーネルと同一の計算
    float dx = state(0) - sensor_x;
    float dy = state(1) - sensor_y;
    float dz = state(2) - sensor_z;
    float vx = state(3);
    float vy = state(4);
    float vz = state(5);

    float range_horiz = std::sqrt(dx * dx + dy * dy);
    float range = std::sqrt(dx * dx + dy * dy + dz * dz);
    float azimuth = std::atan2(dy, dx);
    float elevation = (range_horiz > 1e-6f) ? std::atan2(dz, range_horiz) : 0.0f;
    float doppler = (range > 1e-6f) ? ((dx * vx + dy * vy + dz * vz) / range) : 0.0f;

    MeasVector pred;
    pred << range, azimuth, elevation, doppler;
    return pred;
}

float JPDAAssociation::computeNormalizedDist(const MeasVector& pred_meas,
                                               const Measurement& meas) const
{
    // GPU版 computeMahalanobisCostMatrix と同一の計算
    float innovation[MEAS_DIM];
    innovation[0] = meas.range - pred_meas(0);
    innovation[1] = meas.azimuth - pred_meas(1);
    innovation[2] = meas.elevation - pred_meas(2);
    innovation[3] = meas.doppler - pred_meas(3);

    // 角度の正規化
    if (innovation[1] > static_cast<float>(M_PI))  innovation[1] -= 2.0f * static_cast<float>(M_PI);
    if (innovation[1] < -static_cast<float>(M_PI)) innovation[1] += 2.0f * static_cast<float>(M_PI);

    float noise_stds[MEAS_DIM] = {
        meas_noise_.range_noise,
        meas_noise_.azimuth_noise,
        meas_noise_.elevation_noise,
        meas_noise_.doppler_noise
    };

    float dist_sq = 0.0f;
    for (int i = 0; i < MEAS_DIM; i++) {
        float n = innovation[i] / noise_stds[i];
        dist_sq += n * n;
    }
    return std::sqrt(dist_sq);
}

JPDAResult JPDAAssociation::associate(const std::vector<Track>& tracks,
                                       const std::vector<Measurement>& measurements,
                                       float sensor_x, float sensor_y, float sensor_z)
{
    int num_tracks = static_cast<int>(tracks.size());
    int num_meas = static_cast<int>(measurements.size());

    JPDAResult result;
    result.base_result.track_to_meas.resize(num_tracks, -1);
    result.base_result.meas_to_track.resize(num_meas, -1);
    result.track_updates.resize(num_tracks);

    // 初期化
    for (int i = 0; i < num_tracks; i++) {
        result.track_updates[i].track_index = i;
        result.track_updates[i].has_gated_meas = false;
        result.track_updates[i].beta_0 = 1.0f;
        result.track_updates[i].combined_meas = MeasVector::Zero();
        result.track_updates[i].best_meas_index = -1;
    }

    if (num_tracks == 0 || num_meas == 0) {
        for (int i = 0; i < num_tracks; i++)
            result.base_result.unassigned_tracks.push_back(i);
        for (int j = 0; j < num_meas; j++)
            result.base_result.unassigned_measurements.push_back(j);
        return result;
    }

    float Pd = params_.jpda_pd;
    float lambda_c = params_.jpda_clutter_density;

    // 観測ノイズ標準偏差（正規化距離に基づく対角ガウス尤度）
    float noise_stds[MEAS_DIM] = {
        meas_noise_.range_noise,
        meas_noise_.azimuth_noise,
        meas_noise_.elevation_noise,
        meas_noise_.doppler_noise
    };

    // 正規化定数 (4次元対角ガウス)
    float norm_const = 1.0f;
    for (int d = 0; d < MEAS_DIM; d++) {
        norm_const /= (std::sqrt(2.0f * static_cast<float>(M_PI)) * noise_stds[d]);
    }

    // === Phase 1: 予測観測 + ゲーティング + 尤度計算（全トラック一括） ===
    std::vector<MeasVector> pred_meas(num_tracks);
    std::vector<std::vector<int>> gated_meas(num_tracks);
    std::vector<bool> meas_in_any_gate(num_meas, false);

    // 各トラックの尤度情報
    struct TrackLikelihood {
        std::vector<float> likelihoods;   // ゲート内各観測の尤度
    };
    std::vector<TrackLikelihood> tlike(num_tracks);

    // 各観測に対する全トラック尤度合計（クロストラック競合用）
    std::vector<float> meas_total_likelihood(num_meas, 0.0f);

    for (int i = 0; i < num_tracks; i++) {
        pred_meas[i] = predictMeasurement(tracks[i].state, sensor_x, sensor_y, sensor_z);

        for (int j = 0; j < num_meas; j++) {
            float dist = computeNormalizedDist(pred_meas[i], measurements[j]);
            if (dist < params_.jpda_gate) {
                gated_meas[i].push_back(j);
                meas_in_any_gate[j] = true;
            }
        }

        int n_gated = static_cast<int>(gated_meas[i].size());
        if (n_gated == 0) continue;

        tlike[i].likelihoods.resize(n_gated);

        for (int k = 0; k < n_gated; k++) {
            int j = gated_meas[i][k];

            // 正規化距離ベースの尤度（対角ガウス）
            float innovation[MEAS_DIM];
            innovation[0] = measurements[j].range - pred_meas[i](0);
            innovation[1] = measurements[j].azimuth - pred_meas[i](1);
            innovation[2] = measurements[j].elevation - pred_meas[i](2);
            innovation[3] = measurements[j].doppler - pred_meas[i](3);

            if (innovation[1] > static_cast<float>(M_PI))  innovation[1] -= 2.0f * static_cast<float>(M_PI);
            if (innovation[1] < -static_cast<float>(M_PI)) innovation[1] += 2.0f * static_cast<float>(M_PI);

            float maha_sq = 0.0f;
            for (int d = 0; d < MEAS_DIM; d++) {
                float n = innovation[d] / noise_stds[d];
                maha_sq += n * n;
            }

            tlike[i].likelihoods[k] = norm_const * std::exp(-0.5f * maha_sq);

            // クロストラック競合用: 観測jへの全トラック尤度合計
            meas_total_likelihood[j] += tlike[i].likelihoods[k];
        }
    }

    // ゲートに入らなかった観測 → unassigned_measurements（新規トラック初期化用）
    for (int j = 0; j < num_meas; j++) {
        if (!meas_in_any_gate[j]) {
            result.base_result.unassigned_measurements.push_back(j);
        }
    }

    // === Phase 2: クロストラック排他JPDA β重み + 合成観測生成 ===
    for (int i = 0; i < num_tracks; i++) {
        int n_gated = static_cast<int>(gated_meas[i].size());
        if (n_gated == 0) {
            result.base_result.unassigned_tracks.push_back(i);
            result.track_updates[i].has_gated_meas = false;
            result.track_updates[i].beta_0 = 1.0f;
            continue;
        }

        // β重み: measurement-oriented JPDA
        // β_ij = Pd * L_ij / (λ_c + Pd * Σ_k L_kj)
        std::vector<float> betas(n_gated);
        float sum_beta = 0.0f;

        for (int k = 0; k < n_gated; k++) {
            int j = gated_meas[i][k];
            float denom_j = lambda_c + Pd * meas_total_likelihood[j];
            if (denom_j < 1e-30f) denom_j = 1e-30f;
            betas[k] = Pd * tlike[i].likelihoods[k] / denom_j;
            sum_beta += betas[k];
        }

        // β_i0: 未検出確率
        float beta_0 = std::max(0.0f, 1.0f - sum_beta);
        if (beta_0 < 0.01f) {
            float scale = (1.0f - 0.01f) / sum_beta;
            for (int k = 0; k < n_gated; k++) betas[k] *= scale;
            beta_0 = 0.01f;
        }

        result.track_updates[i].beta_0 = beta_0;

        // === β重み付き合成観測（UKFに渡す） ===
        // z_combined = Σ β_j * z_j / Σ β_j  (正規化)
        MeasVector combined = MeasVector::Zero();
        float beta_sum_for_meas = 0.0f;

        // 方位角の加重平均は角度ラッピングが必要
        // 基準として pred_meas を使い、差分で平均を取る
        for (int k = 0; k < n_gated; k++) {
            int j = gated_meas[i][k];

            MeasVector z;
            z(0) = measurements[j].range;
            z(1) = measurements[j].azimuth;
            z(2) = measurements[j].elevation;
            z(3) = measurements[j].doppler;

            // 方位角差分のラッピング
            float az_diff = z(1) - pred_meas[i](1);
            if (az_diff > static_cast<float>(M_PI))  az_diff -= 2.0f * static_cast<float>(M_PI);
            if (az_diff < -static_cast<float>(M_PI)) az_diff += 2.0f * static_cast<float>(M_PI);

            combined(0) += betas[k] * z(0);
            combined(1) += betas[k] * az_diff;  // 差分で蓄積
            combined(2) += betas[k] * z(2);
            combined(3) += betas[k] * z(3);
            beta_sum_for_meas += betas[k];
        }

        if (beta_sum_for_meas > 1e-10f) {
            combined /= beta_sum_for_meas;
            // 方位角を絶対値に戻す
            combined(1) += pred_meas[i](1);
            // [-π, π] に正規化
            while (combined(1) > static_cast<float>(M_PI))  combined(1) -= 2.0f * static_cast<float>(M_PI);
            while (combined(1) < -static_cast<float>(M_PI)) combined(1) += 2.0f * static_cast<float>(M_PI);
        } else {
            // フォールバック: 予測観測値をそのまま使用
            combined = pred_meas[i];
        }

        result.track_updates[i].combined_meas = combined;
        result.track_updates[i].has_gated_meas = true;

        // 実質未割当判定
        if (beta_0 > 0.9f) {
            result.base_result.unassigned_tracks.push_back(i);
        }

        // 最大β観測をbase_resultに記録（SNR蓄積・IMM更新用）
        int best_k = -1;
        float best_beta = 0.0f;
        for (int k = 0; k < n_gated; k++) {
            if (betas[k] > best_beta) {
                best_beta = betas[k];
                best_k = k;
            }
        }
        if (best_k >= 0 && best_beta > beta_0) {
            int best_j = gated_meas[i][best_k];
            result.base_result.track_to_meas[i] = best_j;
            result.base_result.meas_to_track[best_j] = i;
            result.track_updates[i].best_meas_index = best_j;
        }
    }

    return result;
}

} // namespace fasttracker
