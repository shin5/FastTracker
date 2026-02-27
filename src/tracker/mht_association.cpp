#include "tracker/mht_association.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <functional>
#include <set>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

MHTAssociation::MHTAssociation(const AssociationParams& params,
                                 const MeasurementNoise& meas_noise)
    : params_(params), meas_noise_(meas_noise)
{
}

// ========================================
// 予測観測値（PMBMと同一）
// ========================================
MeasVector MHTAssociation::predictMeasurement(const StateVector& state,
                                               float sensor_x, float sensor_y, float sensor_z) const
{
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

// ========================================
// 正規化距離（PMBMと同一）
// ========================================
float MHTAssociation::computeNormalizedDist(const MeasVector& pred_meas,
                                             const Measurement& meas) const
{
    float innovation[MEAS_DIM];
    innovation[0] = meas.range - pred_meas(0);
    innovation[1] = meas.azimuth - pred_meas(1);
    innovation[2] = meas.elevation - pred_meas(2);
    innovation[3] = meas.doppler - pred_meas(3);

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

// ========================================
// スタンドアロンHungarian法（PMBMと同一）
// ========================================
std::vector<int> MHTAssociation::hungarianSolve(
    const std::vector<float>& cost_matrix, int n, int m)
{
    std::vector<int> assignments(n, -1);
    if (n == 0 || m == 0) return assignments;

    const int N = std::max(n, m);

    std::vector<float> C(N * N, 0.0f);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i * N + j] = cost_matrix[i * m + j];

    std::vector<int> star_col(N, -1);
    std::vector<int> star_row(N, -1);
    std::vector<int> prime_col(N, -1);
    std::vector<bool> row_cover(N, false);
    std::vector<bool> col_cover(N, false);

    for (int i = 0; i < N; i++) {
        float row_min = C[i * N];
        for (int j = 1; j < N; j++)
            if (C[i * N + j] < row_min) row_min = C[i * N + j];
        for (int j = 0; j < N; j++)
            C[i * N + j] -= row_min;
    }

    for (int j = 0; j < N; j++) {
        float col_min = C[j];
        for (int i = 1; i < N; i++)
            if (C[i * N + j] < col_min) col_min = C[i * N + j];
        for (int i = 0; i < N; i++)
            C[i * N + j] -= col_min;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] == 0.0f && star_col[i] == -1 && star_row[j] == -1) {
                star_col[i] = j;
                star_row[j] = i;
            }
        }
    }

    const float HUNGARIAN_EPS = 1e-9f;

    for (;;) {
        std::fill(col_cover.begin(), col_cover.end(), false);
        int covered = 0;
        for (int i = 0; i < N; i++) {
            if (star_col[i] >= 0) {
                col_cover[star_col[i]] = true;
                covered++;
            }
        }
        if (covered >= N) break;

        for (;;) {
            int pr = -1, pc = -1;
            int step46_iter = 0;
            const int MAX_STEP46_ITER = N * N * 4 + 16;
            for (;;) {
                if (++step46_iter > MAX_STEP46_ITER) { pr = -2; break; }
                pr = -1; pc = -1;
                for (int i = 0; i < N && pr == -1; i++) {
                    if (row_cover[i]) continue;
                    for (int j = 0; j < N; j++) {
                        if (!col_cover[j] && C[i * N + j] <= HUNGARIAN_EPS) {
                            pr = i; pc = j;
                            break;
                        }
                    }
                }
                if (pr == -1) {
                    float min_val = std::numeric_limits<float>::max();
                    for (int i = 0; i < N; i++) {
                        if (row_cover[i]) continue;
                        for (int j = 0; j < N; j++) {
                            if (!col_cover[j] && C[i * N + j] < min_val)
                                min_val = C[i * N + j];
                        }
                    }
                    if (min_val <= HUNGARIAN_EPS ||
                        min_val >= std::numeric_limits<float>::max() / 2)
                        goto extract_results;
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            if (row_cover[i]) C[i * N + j] += min_val;
                            if (!col_cover[j]) C[i * N + j] -= min_val;
                        }
                    }
                    continue;
                }
                break;
            }
            if (pr < 0) goto extract_results;

            prime_col[pr] = pc;

            if (star_col[pr] == -1) {
                int path_row = pr, path_col = pc;
                for (;;) {
                    int sr = star_row[path_col];
                    if (sr >= 0) star_col[sr] = -1;
                    star_col[path_row] = path_col;
                    star_row[path_col] = path_row;
                    if (sr < 0) break;
                    path_col = prime_col[sr];
                    path_row = sr;
                }
                std::fill(prime_col.begin(), prime_col.end(), -1);
                std::fill(row_cover.begin(), row_cover.end(), false);
                std::fill(col_cover.begin(), col_cover.end(), false);
                break;
            } else {
                col_cover[star_col[pr]] = false;
                row_cover[pr] = true;
            }
        }
    }
    extract_results:

    for (int i = 0; i < n; i++) {
        int j = star_col[i];
        if (j >= 0 && j < m && cost_matrix[i * m + j] < 1e9f) {
            assignments[i] = j;
        }
    }

    return assignments;
}

// ========================================
// Murty's K-best割当（PMBMと同一）
// ========================================
std::vector<MHTAssociation::Assignment> MHTAssociation::murtyKBest(
    const std::vector<float>& cost_matrix,
    int num_tracks, int num_cols, int K)
{
    std::vector<Assignment> results;
    if (num_tracks == 0 || num_cols == 0 || K <= 0) return results;

    const float BIG_COST = 1e6f;

    std::vector<int> best_assign = hungarianSolve(cost_matrix, num_tracks, num_cols);

    float best_cost = 0.0f;
    for (int i = 0; i < num_tracks; i++) {
        if (best_assign[i] >= 0) {
            best_cost += cost_matrix[i * num_cols + best_assign[i]];
        }
    }

    Assignment best;
    best.track_to_meas = best_assign;
    best.total_cost = best_cost;
    results.push_back(best);

    if (K <= 1) return results;

    struct QueueEntry {
        std::vector<float> cost_mat;
        float solved_cost;
        std::vector<int> assignment;

        bool operator>(const QueueEntry& other) const {
            return solved_cost > other.solved_cost;
        }
    };

    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> pq;

    for (int i = 0; i < num_tracks; i++) {
        int j_forbidden = best_assign[i];
        if (j_forbidden < 0) continue;

        std::vector<float> new_cost = cost_matrix;
        new_cost[i * num_cols + j_forbidden] = BIG_COST;

        for (int ii = 0; ii < i; ii++) {
            int j_fixed = best_assign[ii];
            if (j_fixed < 0) continue;
            for (int jj = 0; jj < num_cols; jj++) {
                if (jj != j_fixed) {
                    new_cost[ii * num_cols + jj] = BIG_COST;
                }
            }
            for (int ii2 = 0; ii2 < num_tracks; ii2++) {
                if (ii2 != ii) {
                    new_cost[ii2 * num_cols + j_fixed] = BIG_COST;
                }
            }
        }

        std::vector<int> sub_assign = hungarianSolve(new_cost, num_tracks, num_cols);
        float sub_cost = 0.0f;
        bool valid = true;
        for (int ii = 0; ii < num_tracks; ii++) {
            if (sub_assign[ii] >= 0) {
                float c = cost_matrix[ii * num_cols + sub_assign[ii]];
                if (c >= BIG_COST * 0.5f) { valid = false; break; }
                sub_cost += c;
            }
        }

        if (valid) {
            QueueEntry entry;
            entry.cost_mat = new_cost;
            entry.solved_cost = sub_cost;
            entry.assignment = sub_assign;
            pq.push(entry);
        }
    }

    while (!pq.empty() && static_cast<int>(results.size()) < K) {
        QueueEntry top = pq.top();
        pq.pop();

        bool duplicate = false;
        for (const auto& existing : results) {
            if (existing.track_to_meas == top.assignment) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        Assignment a;
        a.track_to_meas = top.assignment;
        a.total_cost = top.solved_cost;
        results.push_back(a);

        for (int i = 0; i < num_tracks; i++) {
            int j_forbidden = top.assignment[i];
            if (j_forbidden < 0) continue;

            std::vector<float> new_cost = top.cost_mat;
            new_cost[i * num_cols + j_forbidden] = BIG_COST;

            for (int ii = 0; ii < i; ii++) {
                int j_fixed = top.assignment[ii];
                if (j_fixed < 0) continue;
                for (int jj = 0; jj < num_cols; jj++) {
                    if (jj != j_fixed) {
                        new_cost[ii * num_cols + jj] = BIG_COST;
                    }
                }
                for (int ii2 = 0; ii2 < num_tracks; ii2++) {
                    if (ii2 != ii) {
                        new_cost[ii2 * num_cols + j_fixed] = BIG_COST;
                    }
                }
            }

            std::vector<int> sub_assign = hungarianSolve(new_cost, num_tracks, num_cols);
            float sub_cost = 0.0f;
            bool valid = true;
            for (int ii = 0; ii < num_tracks; ii++) {
                if (sub_assign[ii] >= 0) {
                    float c = cost_matrix[ii * num_cols + sub_assign[ii]];
                    if (c >= BIG_COST * 0.5f) { valid = false; break; }
                    sub_cost += c;
                }
            }

            if (valid) {
                QueueEntry entry;
                entry.cost_mat = new_cost;
                entry.solved_cost = sub_cost;
                entry.assignment = sub_assign;
                pq.push(entry);
            }
        }
    }

    return results;
}

// ========================================
// MHTアソシエーション メイン
// ========================================
MHTResult MHTAssociation::associate(const std::vector<Track>& tracks,
                                     const std::vector<Measurement>& measurements,
                                     float sensor_x, float sensor_y, float sensor_z)
{
    int num_tracks = static_cast<int>(tracks.size());
    int num_meas = static_cast<int>(measurements.size());

    MHTResult result;
    result.base_result.track_to_meas.resize(num_tracks, -1);
    result.base_result.meas_to_track.resize(num_meas, -1);
    result.track_info.resize(num_tracks);

    // 初期化
    for (int i = 0; i < num_tracks; i++) {
        result.track_info[i].track_index = i;
        result.track_info[i].best_meas_index = -1;
        result.track_info[i].assignment_confidence = 0.0f;
        result.track_info[i].miss_prob = 1.0f;
        result.track_info[i].existence_prob = tracks[i].existence_prob;
    }

    if (num_tracks == 0 || num_meas == 0) {
        for (int i = 0; i < num_tracks; i++)
            result.base_result.unassigned_tracks.push_back(i);
        for (int j = 0; j < num_meas; j++)
            result.base_result.unassigned_measurements.push_back(j);
        return result;
    }

    float Pd = params_.mht_pd;
    float lambda_c = params_.mht_clutter_density;
    int K = params_.mht_k_best;
    int M = params_.mht_max_hypotheses;
    float decay = params_.mht_score_decay;
    float prune_ratio = params_.mht_prune_ratio;

    // === Phase 1: 予測観測 + ゲーティング + 距離・尤度計算 ===
    // （PMBMと同一）
    float noise_stds[MEAS_DIM] = {
        meas_noise_.range_noise,
        meas_noise_.azimuth_noise,
        meas_noise_.elevation_noise,
        meas_noise_.doppler_noise
    };

    float norm_const = 1.0f;
    for (int d = 0; d < MEAS_DIM; d++) {
        norm_const /= (std::sqrt(2.0f * static_cast<float>(M_PI)) * noise_stds[d]);
    }

    std::vector<MeasVector> pred_meas(num_tracks);
    std::vector<std::vector<int>> gated_meas(num_tracks);
    std::vector<bool> meas_in_any_gate(num_meas, false);

    std::vector<std::vector<float>> maha_sq_matrix(num_tracks, std::vector<float>(num_meas, 0.0f));
    std::vector<std::vector<float>> log_likelihood(num_tracks, std::vector<float>(num_meas, -1e30f));

    for (int i = 0; i < num_tracks; i++) {
        pred_meas[i] = predictMeasurement(tracks[i].state, sensor_x, sensor_y, sensor_z);

        float effective_gate = params_.mht_gate;
        if (tracks[i].track_state == TrackState::TENTATIVE) {
            effective_gate = std::min(params_.mht_gate, 40.0f);
        }

        for (int j = 0; j < num_meas; j++) {
            float dist = computeNormalizedDist(pred_meas[i], measurements[j]);
            if (dist < effective_gate) {
                gated_meas[i].push_back(j);
                meas_in_any_gate[j] = true;

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

                maha_sq_matrix[i][j] = maha_sq;
                float L_ij = norm_const * std::exp(-0.5f * maha_sq);
                float detection_lr = Pd * L_ij / lambda_c;
                if (detection_lr > 1e-30f) {
                    log_likelihood[i][j] = std::log(detection_lr);
                }
            }
        }
    }

    // ゲート外観測 → unassigned
    for (int j = 0; j < num_meas; j++) {
        if (!meas_in_any_gate[j]) {
            result.base_result.unassigned_measurements.push_back(j);
        }
    }

    // === Phase 2: 拡張コスト行列構築 ===
    // （PMBMと同一）
    int num_cols = num_meas + num_tracks;
    const float BIG_COST = 1e6f;
    float gate_sq = params_.mht_gate * params_.mht_gate;

    std::vector<float> ext_cost(num_tracks * num_cols, BIG_COST);

    for (int i = 0; i < num_tracks; i++) {
        for (int j : gated_meas[i]) {
            ext_cost[i * num_cols + j] = maha_sq_matrix[i][j];
        }

        float miss_cost;
        if (tracks[i].track_state == TrackState::CONFIRMED) {
            miss_cost = gated_meas[i].empty() ? gate_sq : BIG_COST * 0.5f;
        } else {
            float r_i = tracks[i].existence_prob;
            float effective_r = std::max(r_i, 0.6f);
            miss_cost = effective_r * gate_sq;
        }
        ext_cost[i * num_cols + num_meas + i] = miss_cost;
    }

    // === Phase 3: Murty's K-best割当 ===
    std::vector<Assignment> k_best = murtyKBest(ext_cost, num_tracks, num_cols, K);

    if (k_best.empty()) {
        for (int i = 0; i < num_tracks; i++)
            result.base_result.unassigned_tracks.push_back(i);
        return result;
    }

    // === Phase 4: 多フレーム仮説結合（MHT固有） ===

    // 4a. 現フレームの各K-best割当をtrack_idキーに変換し、スコアを計算
    float log_miss = std::log(std::max(1.0f - Pd, 1e-10f));

    struct CurrentAssignment {
        std::unordered_map<int, int> track_to_meas;  // track_id → meas_index (-1=miss)
        std::unordered_map<int, std::array<float, 4>> track_to_meas_pos;  // track_id → [r,az,el,dop]
        float score;  // 現フレーム対数尤度
    };
    std::vector<CurrentAssignment> current_assignments(k_best.size());

    for (size_t k = 0; k < k_best.size(); k++) {
        float total_score = 0.0f;
        for (int i = 0; i < num_tracks; i++) {
            int col = k_best[k].track_to_meas[i];
            int meas_idx = (col >= 0 && col < num_meas) ? col : -1;
            int tid = tracks[i].id;
            current_assignments[k].track_to_meas[tid] = meas_idx;

            if (meas_idx >= 0) {
                total_score += log_likelihood[i][meas_idx];
                current_assignments[k].track_to_meas_pos[tid] = {
                    measurements[meas_idx].range,
                    measurements[meas_idx].azimuth,
                    measurements[meas_idx].elevation,
                    measurements[meas_idx].doppler
                };
            }
        }
        current_assignments[k].score = total_score;
    }

    // 4b. 仮説結合（位置ベース一貫性スコア付き）
    float switch_cost = params_.mht_switch_cost;

    if (frame_count_ == 0 || hypotheses_.empty()) {
        // 初回フレーム: K-best割当をそのままグローバル仮説化
        hypotheses_.clear();
        for (size_t k = 0; k < current_assignments.size(); k++) {
            MHTGlobalHypothesis h;
            h.id = next_hyp_id_++;
            h.accumulated_score = current_assignments[k].score;
            h.track_to_meas = current_assignments[k].track_to_meas;
            h.track_to_meas_pos = current_assignments[k].track_to_meas_pos;
            hypotheses_.push_back(h);
        }
    } else {
        // 前フレームのM仮説 × 現フレームのK割当 = M×K候補
        std::vector<MHTGlobalHypothesis> candidates;
        candidates.reserve(hypotheses_.size() * current_assignments.size());

        // 一貫性判定の閾値: この正規化距離以下なら同一目標と判定
        const float consistency_gate = 8.0f;

        for (const auto& prev_hyp : hypotheses_) {
            for (size_t k = 0; k < current_assignments.size(); k++) {
                // 位置ベース割当一貫性スコア:
                // 前フレームの観測位置と現フレームの観測位置を比較し、
                // 空間的に近い（同一目標）なら+bonus、遠い（別目標）なら-penalty
                float consistency = 0.0f;
                if (switch_cost > 0.0f) {
                    for (const auto& [tid, curr_meas] : current_assignments[k].track_to_meas) {
                        if (curr_meas < 0) continue;  // missはニュートラル

                        auto prev_pos_it = prev_hyp.track_to_meas_pos.find(tid);
                        if (prev_pos_it == prev_hyp.track_to_meas_pos.end()) continue;  // 前フレームmissもニュートラル

                        // 前フレームの観測位置と現フレームの観測位置の正規化距離
                        const auto& pp = prev_pos_it->second;
                        float dr = (measurements[curr_meas].range    - pp[0]) / noise_stds[0];
                        float da = (measurements[curr_meas].azimuth  - pp[1]) / noise_stds[1];
                        float de = (measurements[curr_meas].elevation- pp[2]) / noise_stds[2];
                        float dd = (measurements[curr_meas].doppler  - pp[3]) / noise_stds[3];

                        // 方位角ラッピング
                        if (da > static_cast<float>(M_PI) / noise_stds[1])
                            da -= 2.0f * static_cast<float>(M_PI) / noise_stds[1];
                        if (da < -static_cast<float>(M_PI) / noise_stds[1])
                            da += 2.0f * static_cast<float>(M_PI) / noise_stds[1];

                        float dist = std::sqrt(dr * dr + da * da + de * de + dd * dd);

                        if (dist < consistency_gate) {
                            // 同一目標の観測 → 一貫性ボーナス
                            consistency += switch_cost;
                        } else {
                            // 別目標の観測 → 切替ペナルティ
                            consistency -= switch_cost;
                        }
                    }
                }

                MHTGlobalHypothesis new_hyp;
                new_hyp.id = next_hyp_id_++;
                new_hyp.accumulated_score = prev_hyp.accumulated_score * decay
                                          + current_assignments[k].score + consistency;
                new_hyp.track_to_meas = current_assignments[k].track_to_meas;
                new_hyp.track_to_meas_pos = current_assignments[k].track_to_meas_pos;
                candidates.push_back(std::move(new_hyp));
            }
        }

        // 累積スコア降順でソート
        std::sort(candidates.begin(), candidates.end(),
            [](const MHTGlobalHypothesis& a, const MHTGlobalHypothesis& b) {
                return a.accumulated_score > b.accumulated_score;
            });

        // 重複除去（同一current assignment, 高スコア側を保持）
        // assignment比較にはtrack_to_measマップを使用
        hypotheses_.clear();
        for (auto& cand : candidates) {
            if (static_cast<int>(hypotheses_.size()) >= M) break;

            // プルーニング: ベストスコアとの差がlog(1/prune_ratio)を超えたら打ち切り
            if (!hypotheses_.empty()) {
                float score_diff = hypotheses_[0].accumulated_score - cand.accumulated_score;
                if (prune_ratio > 0.0f && score_diff > -std::log(prune_ratio)) {
                    break;
                }
            }

            // 重複チェック: 同一割当の仮説は最高スコアのみ保持
            bool duplicate = false;
            for (const auto& existing : hypotheses_) {
                if (existing.track_to_meas == cand.track_to_meas) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                hypotheses_.push_back(std::move(cand));
            }
        }
    }

    frame_count_++;

    // === Phase 5: トラックリスト変更の処理 ===

    // 現在のトラックIDセットを構築
    std::set<int> current_track_ids;
    for (int i = 0; i < num_tracks; i++) {
        current_track_ids.insert(tracks[i].id);
    }

    for (auto& hyp : hypotheses_) {
        // 新規トラック（仮説に未登録）: ベストK-bestの割当をデフォルトとして追加
        for (int i = 0; i < num_tracks; i++) {
            int tid = tracks[i].id;
            if (hyp.track_to_meas.find(tid) == hyp.track_to_meas.end()) {
                // ベスト割当（k_best[0]）から取得
                int col = k_best[0].track_to_meas[i];
                int meas_idx = (col >= 0 && col < num_meas) ? col : -1;
                hyp.track_to_meas[tid] = meas_idx;
                if (meas_idx >= 0) {
                    hyp.track_to_meas_pos[tid] = {
                        measurements[meas_idx].range,
                        measurements[meas_idx].azimuth,
                        measurements[meas_idx].elevation,
                        measurements[meas_idx].doppler
                    };
                }
            }
        }

        // 削除済みトラック（仮説に残存）: 除去
        auto it = hyp.track_to_meas.begin();
        while (it != hyp.track_to_meas.end()) {
            if (current_track_ids.find(it->first) == current_track_ids.end()) {
                hyp.track_to_meas_pos.erase(it->first);
                it = hyp.track_to_meas.erase(it);
            } else {
                ++it;
            }
        }
    }

    // === Phase 6: スコア加重マージナル確率 + 存在確率更新 ===
    if (!hypotheses_.empty()) {
        // 6a. 仮説のスコアを正規化重みに変換（ソフトマックス）
        float max_score = hypotheses_[0].accumulated_score;
        for (size_t h = 1; h < hypotheses_.size(); h++) {
            if (hypotheses_[h].accumulated_score > max_score)
                max_score = hypotheses_[h].accumulated_score;
        }

        std::vector<float> hyp_weights(hypotheses_.size());
        float weight_sum = 0.0f;
        for (size_t h = 0; h < hypotheses_.size(); h++) {
            hyp_weights[h] = std::exp(hypotheses_[h].accumulated_score - max_score);
            weight_sum += hyp_weights[h];
        }
        if (weight_sum < 1e-30f) weight_sum = 1e-30f;
        for (size_t h = 0; h < hypotheses_.size(); h++) {
            hyp_weights[h] /= weight_sum;
        }

        // 6b. 各トラックのマージナルmiss確率と割当信頼度を計算
        float Ps = 0.999f;  // 生存確率

        for (int i = 0; i < num_tracks; i++) {
            int tid = tracks[i].id;
            auto it0 = hypotheses_[0].track_to_meas.find(tid);
            int best_meas = (it0 != hypotheses_[0].track_to_meas.end()) ? it0->second : -1;

            float miss_weight = 0.0f;
            float agree_weight = 0.0f;

            for (size_t h = 0; h < hypotheses_.size(); h++) {
                auto hit = hypotheses_[h].track_to_meas.find(tid);
                int hyp_meas = (hit != hypotheses_[h].track_to_meas.end()) ? hit->second : -1;

                if (hyp_meas < 0 || hyp_meas >= num_meas) {
                    miss_weight += hyp_weights[h];
                }
                if (hyp_meas == best_meas) {
                    agree_weight += hyp_weights[h];
                }
            }

            result.track_info[i].best_meas_index = best_meas;
            result.track_info[i].assignment_confidence = agree_weight;
            result.track_info[i].miss_prob = miss_weight;

            // 6c. ベイジアン存在確率更新（PMBMと同一の公式）
            float r_i = tracks[i].existence_prob;
            float denom = 1.0f - r_i * Pd;
            if (denom < 1e-10f) denom = 1e-10f;
            float r_miss = r_i * (1.0f - Pd) / denom;

            float r_new = miss_weight * r_miss + (1.0f - miss_weight) * 1.0f;
            r_new *= Ps;
            r_new = std::max(0.001f, std::min(0.999f, r_new));

            result.track_info[i].existence_prob = r_new;
        }
    }

    // === Phase 7: ベスト仮説抽出 → base_result ===
    if (!hypotheses_.empty()) {
        const auto& best_hyp = hypotheses_[0];  // 最高累積スコア

        for (int i = 0; i < num_tracks; i++) {
            int tid = tracks[i].id;
            auto it = best_hyp.track_to_meas.find(tid);
            if (it != best_hyp.track_to_meas.end() && it->second >= 0 && it->second < num_meas) {
                int meas_idx = it->second;
                result.base_result.track_to_meas[i] = meas_idx;
                result.base_result.meas_to_track[meas_idx] = i;
                result.track_info[i].best_meas_index = meas_idx;
            }
        }
    }

    // 未割当トラック/観測
    for (int i = 0; i < num_tracks; i++) {
        if (result.base_result.track_to_meas[i] < 0) {
            result.base_result.unassigned_tracks.push_back(i);
        }
    }

    for (int j = 0; j < num_meas; j++) {
        if (result.base_result.meas_to_track[j] < 0 && meas_in_any_gate[j]) {
            result.base_result.unassigned_measurements.push_back(j);
        }
    }

    return result;
}

} // namespace fasttracker
