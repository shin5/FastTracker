#include "tracker/glmb_association.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

GLMBAssociation::GLMBAssociation(const AssociationParams& params,
                                   const MeasurementNoise& meas_noise)
    : params_(params), meas_noise_(meas_noise)
{
}

// ========================================
// 予測観測値（MHT/PMBMと同一）
// ========================================
MeasVector GLMBAssociation::predictMeasurement(const StateVector& state,
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
// 正規化距離（MHT/PMBMと同一）
// ========================================
float GLMBAssociation::computeNormalizedDist(const MeasVector& pred_meas,
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
// スタンドアロンHungarian法（MHT/PMBMと同一）
// ========================================
std::vector<int> GLMBAssociation::hungarianSolve(
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
// Murty's K-best割当（MHT/PMBMと同一）
// ========================================
std::vector<GLMBAssociation::Assignment> GLMBAssociation::murtyKBest(
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
// Gibbs Sampler（Murty's K-bestの代替）
// ========================================
std::vector<GLMBAssociation::Assignment> GLMBAssociation::gibbsSample(
    const std::vector<float>& cost_matrix,
    int num_tracks, int num_cols, int K)
{
    std::vector<Assignment> results;
    if (num_tracks == 0 || num_cols == 0 || K <= 0) return results;

    const int num_meas = num_cols - num_tracks;
    const int N_sweeps = params_.glmb_gibbs_sweeps;
    const int burnin = params_.glmb_gibbs_burnin;
    const float BIG_COST = 1e6f;

    // Step 1: Hungarian解で初期化（全スイープで使用）
    std::vector<int> current_assign = hungarianSolve(cost_matrix, num_tracks, num_cols);

    // Hungarian解を保存（結果に必ず含める保証用）
    std::vector<int> hungarian_assign = current_assign;

    // 観測列の排他制約管理
    std::vector<bool> col_taken(num_meas, false);
    for (int i = 0; i < num_tracks; i++) {
        int j = current_assign[i];
        if (j >= 0 && j < num_meas) {
            col_taken[j] = true;
        }
    }

    std::vector<int> track_order(num_tracks);
    std::iota(track_order.begin(), track_order.end(), 0);
    std::vector<std::vector<int>> samples;
    samples.reserve(N_sweeps - burnin);
    std::vector<float> probs(num_cols);

    // Temperature annealing: T linearly decreases from T_max to T_min
    // Early sweeps (T high): diverse exploration of assignment space
    // Late sweeps (T low): concentrate sampling around optimal assignments
    // This reduces variance in recorded samples while maintaining exploration during burn-in
    const float T_max = 1.0f;
    const float T_min = 0.2f;

    for (int s = 0; s < N_sweeps; s++) {
        // Linear annealing schedule
        float T = (N_sweeps > 1)
            ? T_max - (T_max - T_min) * static_cast<float>(s) / static_cast<float>(N_sweeps - 1)
            : T_min;

        std::shuffle(track_order.begin(), track_order.end(), rng_);

        for (int idx = 0; idx < num_tracks; idx++) {
            int i = track_order[idx];

            int old_j = current_assign[i];
            if (old_j >= 0 && old_j < num_meas) {
                col_taken[old_j] = false;
            }

            std::fill(probs.begin(), probs.end(), 0.0f);
            for (int j = 0; j < num_cols; j++) {
                float cost = cost_matrix[i * num_cols + j];
                if (cost >= BIG_COST * 0.5f) continue;

                if (j < num_meas) {
                    if (col_taken[j]) continue;
                    probs[j] = std::exp(-cost / T);
                } else {
                    if (j == num_meas + i) {
                        probs[j] = std::exp(-cost / T);
                    }
                }
            }

            float sum = 0.0f;
            for (int j = 0; j < num_cols; j++) sum += probs[j];

            if (sum < 1e-30f) {
                current_assign[i] = num_meas + i;
            } else {
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int sampled_j = dist(rng_);
                current_assign[i] = sampled_j;
                if (sampled_j >= 0 && sampled_j < num_meas) {
                    col_taken[sampled_j] = true;
                }
            }
        }

        if (s >= burnin) {
            samples.push_back(current_assign);
        }
    }

    // Step 3: コスト計算、重複除去、上位K個返却
    std::vector<Assignment> all_assignments;
    all_assignments.reserve(samples.size());

    for (const auto& sample : samples) {
        float total_cost = 0.0f;
        bool valid = true;
        for (int i = 0; i < num_tracks; i++) {
            int j = sample[i];
            if (j >= 0 && j < num_cols) {
                float c = cost_matrix[i * num_cols + j];
                if (c >= BIG_COST * 0.5f) { valid = false; break; }
                total_cost += c;
            }
        }
        if (!valid) continue;

        Assignment a;
        a.track_to_meas = sample;
        a.total_cost = total_cost;
        all_assignments.push_back(a);
    }

    std::sort(all_assignments.begin(), all_assignments.end(),
        [](const Assignment& a, const Assignment& b) {
            return a.total_cost < b.total_cost;
        });

    // 重複除去、上位K個
    for (const auto& a : all_assignments) {
        if (static_cast<int>(results.size()) >= K) break;
        bool dup = false;
        for (const auto& existing : results) {
            if (existing.track_to_meas == a.track_to_meas) { dup = true; break; }
        }
        if (!dup) {
            results.push_back(a);
        }
    }

    // Hungarian解を必ず結果に含める（Gibbs探索がmiss偏重に陥った場合の安全弁）
    // 決定論的最適解が常に仮説候補にあることで、ビームステアリング連鎖劣化を防止
    {
        float h_cost = 0.0f;
        bool valid = true;
        for (int i = 0; i < num_tracks; i++) {
            if (hungarian_assign[i] >= 0) {
                float c = cost_matrix[i * num_cols + hungarian_assign[i]];
                if (c >= BIG_COST * 0.5f) { valid = false; break; }
                h_cost += c;
            }
        }
        if (valid) {
            bool dup = false;
            for (const auto& existing : results) {
                if (existing.track_to_meas == hungarian_assign) { dup = true; break; }
            }
            if (!dup) {
                Assignment a;
                a.track_to_meas = hungarian_assign;
                a.total_cost = h_cost;
                // Insert at correct position by cost
                auto pos = std::lower_bound(results.begin(), results.end(), a,
                    [](const Assignment& x, const Assignment& y) {
                        return x.total_cost < y.total_cost;
                    });
                results.insert(pos, a);
            }
        }
    }

    if (results.empty()) {
        Assignment a;
        a.track_to_meas = hungarian_assign;
        float h_cost = 0.0f;
        for (int i = 0; i < num_tracks; i++) {
            if (hungarian_assign[i] >= 0)
                h_cost += cost_matrix[i * num_cols + hungarian_assign[i]];
        }
        a.total_cost = h_cost;
        results.push_back(a);
    }

    return results;
}

// ========================================
// 仮説生成ディスパッチ（Murty / Gibbs 切替）
// ========================================
std::vector<GLMBAssociation::Assignment> GLMBAssociation::generateAssignments(
    const std::vector<float>& cost_matrix,
    int num_tracks, int num_cols, int K)
{
    if (params_.glmb_sampler == GLMBSampler::GIBBS) {
        return gibbsSample(cost_matrix, num_tracks, num_cols, K);
    }
    return murtyKBest(cost_matrix, num_tracks, num_cols, K);
}

// ========================================
// カーディナリティ分布計算（GLMB固有）
// ========================================
std::vector<float> GLMBAssociation::computeCardinalityDistribution(
    const std::vector<GLMBGlobalHypothesis>& hyps,
    const std::vector<float>& weights,
    int max_n) const
{
    std::vector<float> card_dist(max_n + 1, 0.0f);
    for (size_t k = 0; k < hyps.size(); k++) {
        int n = static_cast<int>(hyps[k].active_labels.size());
        if (n <= max_n) {
            card_dist[n] += weights[k];
        }
    }
    return card_dist;
}

// ========================================
// GLMBアソシエーション メイン
// ========================================
GLMBResult GLMBAssociation::associate(const std::vector<Track>& tracks,
                                       const std::vector<Measurement>& measurements,
                                       float sensor_x, float sensor_y, float sensor_z)
{
    int num_tracks = static_cast<int>(tracks.size());
    int num_meas = static_cast<int>(measurements.size());

    GLMBResult result;
    result.base_result.track_to_meas.resize(num_tracks, -1);
    result.base_result.meas_to_track.resize(num_meas, -1);
    result.track_updates.resize(num_tracks);
    result.estimated_num_targets = 0.0f;

    // 初期化
    for (int i = 0; i < num_tracks; i++) {
        result.track_updates[i].track_index = i;
        result.track_updates[i].has_gated_meas = false;
        result.track_updates[i].existence_prob = tracks[i].existence_prob;
        result.track_updates[i].miss_prob = 1.0f;
        result.track_updates[i].best_meas_index = -1;
        result.track_updates[i].best_meas_prob = 0.0f;
        result.track_updates[i].assignment_confidence = 0.0f;
    }

    if (num_tracks == 0 || num_meas == 0) {
        for (int i = 0; i < num_tracks; i++)
            result.base_result.unassigned_tracks.push_back(i);
        for (int j = 0; j < num_meas; j++)
            result.base_result.unassigned_measurements.push_back(j);
        return result;
    }

    float Pd = params_.glmb_pd;
    float lambda_c_config = params_.glmb_clutter_density;
    int K = params_.glmb_k_best;
    int M = params_.glmb_max_hypotheses;
    float decay = params_.glmb_score_decay;
    float Ps = params_.glmb_survival_prob;

    // Adaptive clutter density: estimate from measurement count
    // In heavy-clutter scenarios (pfa=1e-5), configured lambda_c << actual density.
    // Use measurement-based estimate clamped to [lambda_c_config, 0.01]
    float gate_volume = 1.0f;
    for (int d = 0; d < MEAS_DIM; d++) {
        float noise_stds_d[4] = {
            meas_noise_.range_noise,
            meas_noise_.azimuth_noise,
            meas_noise_.elevation_noise,
            meas_noise_.doppler_noise
        };
        gate_volume *= 2.0f * params_.glmb_gate * noise_stds_d[d];
    }
    float lambda_c_est = (num_meas > 0 && gate_volume > 1e-10f)
        ? static_cast<float>(num_meas) / gate_volume : lambda_c_config;
    float lambda_c = std::max(lambda_c_config, std::min(lambda_c_est, 0.01f));

    // === Phase 1: 予測観測 + ゲーティング + 距離・尤度計算 ===
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

        float effective_gate = params_.glmb_gate;
        if (tracks[i].track_state == TrackState::TENTATIVE) {
            // Tighter gate for TENTATIVE tracks to reject clutter
            effective_gate = std::min(params_.glmb_gate, 15.0f);
        } else if (tracks[i].track_state == TrackState::CONFIRMED) {
            // Wider gate for CONFIRMED tracks to maintain tracking
            effective_gate = params_.glmb_gate * 1.5f;
        }

        for (int j = 0; j < num_meas; j++) {
            float dist = computeNormalizedDist(pred_meas[i], measurements[j]);
            if (dist < effective_gate) {
                gated_meas[i].push_back(j);
                meas_in_any_gate[j] = true;
                result.track_updates[i].has_gated_meas = true;

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
                // Cap detection_lr to prevent lambda_c mismatch
                // (configured lambda_c << actual clutter density → every gated
                //  measurement looks like a confident detection)
                // With cap=10, a track needs ~3 consistent detections to reach
                // high existence probability — good for clutter rejection.
                detection_lr = std::min(detection_lr, 10.0f);
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
    int num_cols = num_meas + num_tracks;
    const float BIG_COST = 1e6f;
    float gate_sq = params_.glmb_gate * params_.glmb_gate;

    std::vector<float> ext_cost(num_tracks * num_cols, BIG_COST);

    for (int i = 0; i < num_tracks; i++) {
        for (int j : gated_meas[i]) {
            ext_cost[i * num_cols + j] = maha_sq_matrix[i][j];
        }

        float miss_cost;
        if (tracks[i].track_state == TrackState::CONFIRMED) {
            // Allow confirmed tracks to miss even with gated measurements.
            // In high-clutter, all gated meas may be clutter.
            // Use gate_sq * (1 - Pd) so miss is affordable but not free.
            miss_cost = gate_sq * (1.0f - Pd);
        } else {
            float r_i = tracks[i].existence_prob;
            // Low-existence TENTATIVE tracks: higher miss cost scaled by existence
            // This makes it HARDER for low-quality tentatives to capture measurements
            // (they prefer to miss, leaving measurements for better tracks)
            float effective_r = std::max(r_i, 0.15f);
            miss_cost = effective_r * gate_sq * 0.8f;
        }
        ext_cost[i * num_cols + num_meas + i] = miss_cost;
    }

    // === Phase 3: Murty's K-best割当 ===
    std::vector<Assignment> k_best = generateAssignments(ext_cost, num_tracks, num_cols, K);

    if (k_best.empty()) {
        for (int i = 0; i < num_tracks; i++)
            result.base_result.unassigned_tracks.push_back(i);
        return result;
    }

    // === Phase 4: フレーム間仮説累積（ラベル付き — GLMB固有）===

    float log_miss = std::log(std::max(1.0f - Pd, 1e-10f));

    // 4a. 現フレームの各K-best割当をtrack_idキーに変換し、スコアとラベル集合を計算
    struct CurrentAssignment {
        std::unordered_map<int, int> track_to_meas;
        std::unordered_map<int, std::array<float, 4>> track_to_meas_pos;
        std::set<int> active_labels;  // GLMB: 検出されたラベルの集合
        float score;
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
                // Existence-weighted scoring: favor detection of high-existence tracks
                // High-existence (r>0.7): bonus up to +0.5
                // Low-existence (r<0.3): penalty up to -0.3
                // This makes hypotheses that detect real targets score higher
                float r_i = tracks[i].existence_prob;
                float exist_bonus = (r_i - 0.5f) * 1.0f;  // r=0.9→+0.4, r=0.2→-0.3
                total_score += exist_bonus;

                current_assignments[k].track_to_meas_pos[tid] = {
                    measurements[meas_idx].range,
                    measurements[meas_idx].azimuth,
                    measurements[meas_idx].elevation,
                    measurements[meas_idx].doppler
                };
                current_assignments[k].active_labels.insert(tid);
            } else {
                total_score += log_miss;
                // GLMB: 存在確率が十分高いトラックのみmissでもアクティブラベルに含む
                if (tracks[i].existence_prob > params_.glmb_confirm_existence) {
                    current_assignments[k].active_labels.insert(tid);
                }
            }
        }
        current_assignments[k].score = total_score;
    }

    // 4b. フレーム間仮説累積（ラベル集合付き）
    if (frame_count_ == 0 || hypotheses_.empty()) {
        // 初回フレーム: K-best割当をそのままグローバル仮説化
        hypotheses_.clear();
        for (size_t k = 0; k < current_assignments.size(); k++) {
            GLMBGlobalHypothesis h;
            h.id = next_hyp_id_++;
            h.accumulated_score = current_assignments[k].score;
            h.track_to_meas = current_assignments[k].track_to_meas;
            h.track_to_meas_pos = current_assignments[k].track_to_meas_pos;
            h.active_labels = current_assignments[k].active_labels;
            hypotheses_.push_back(h);
        }
    } else {
        // 前フレームのM仮説 × 現フレームのK割当 = M×K候補
        std::vector<GLMBGlobalHypothesis> candidates;
        candidates.reserve(hypotheses_.size() * current_assignments.size());

        for (const auto& prev_hyp : hypotheses_) {
            for (size_t k = 0; k < current_assignments.size(); k++) {
                GLMBGlobalHypothesis new_hyp;
                new_hyp.id = next_hyp_id_++;
                new_hyp.accumulated_score = prev_hyp.accumulated_score * decay
                                          + current_assignments[k].score;
                new_hyp.track_to_meas = current_assignments[k].track_to_meas;
                new_hyp.track_to_meas_pos = current_assignments[k].track_to_meas_pos;

                // GLMB: ラベル集合の更新
                // 前フレームのラベルと現フレームのラベルの和集合をベースに、
                // 現フレームで検出もmissもされなかったラベル（削除済み）を除外
                new_hyp.active_labels = current_assignments[k].active_labels;
                // 前フレームのラベルで、現フレームのトラックリストにまだ存在するものを保持
                for (int label : prev_hyp.active_labels) {
                    // 現フレームのactive_labelsに既にあればスキップ
                    if (new_hyp.active_labels.count(label) > 0) continue;
                    // 現フレームのtrack_to_measにあるが検出されなかった（低存在確率で除外された）
                    // → 前フレームから引き継がない
                }

                candidates.push_back(std::move(new_hyp));
            }
        }

        // Cardinality-informed scoring: penalize hypotheses with outlier cardinality
        // Compute mode cardinality from top-K candidates (by raw score)
        if (candidates.size() > 3) {
            // Pre-sort by raw score to find top candidates
            std::partial_sort(candidates.begin(),
                candidates.begin() + std::min(candidates.size(), size_t(10)),
                candidates.end(),
                [](const GLMBGlobalHypothesis& a, const GLMBGlobalHypothesis& b) {
                    return a.accumulated_score > b.accumulated_score;
                });
            // Find mode cardinality from top-10
            std::unordered_map<int, int> card_counts;
            int check_n = std::min(candidates.size(), size_t(10));
            for (int ci = 0; ci < check_n; ci++) {
                int card = static_cast<int>(candidates[ci].active_labels.size());
                card_counts[card]++;
            }
            int mode_card = 0, mode_count = 0;
            for (const auto& [card, cnt] : card_counts) {
                if (cnt > mode_count) { mode_count = cnt; mode_card = card; }
            }
            // Apply cardinality penalty: |card - mode| * 0.5 penalty
            for (auto& cand : candidates) {
                int card = static_cast<int>(cand.active_labels.size());
                int card_diff = std::abs(card - mode_card);
                if (card_diff > 0) {
                    cand.accumulated_score -= card_diff * 0.5f;
                }
            }
        }

        // 累積スコア降順でソート (with cardinality penalty applied)
        std::sort(candidates.begin(), candidates.end(),
            [](const GLMBGlobalHypothesis& a, const GLMBGlobalHypothesis& b) {
                return a.accumulated_score > b.accumulated_score;
            });

        // プルーニング + 重複除去
        float prune_threshold = -std::log(std::max(params_.glmb_prune_weight, 1e-10f));
        hypotheses_.clear();
        for (auto& cand : candidates) {
            if (static_cast<int>(hypotheses_.size()) >= M) break;

            // プルーニング: ベストスコアとの差が閾値を超えたら打ち切り
            if (!hypotheses_.empty()) {
                float score_diff = hypotheses_[0].accumulated_score - cand.accumulated_score;
                if (score_diff > prune_threshold) {
                    break;
                }
            }

            // 重複チェック: 同一割当 + 同一ラベル集合の仮説は最高スコアのみ保持
            bool duplicate = false;
            for (const auto& existing : hypotheses_) {
                if (existing.track_to_meas == cand.track_to_meas &&
                    existing.active_labels == cand.active_labels) {
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
    std::set<int> current_track_ids;
    for (int i = 0; i < num_tracks; i++) {
        current_track_ids.insert(tracks[i].id);
    }

    for (auto& hyp : hypotheses_) {
        // 新規トラック: ベストK-bestの割当をデフォルトとして追加
        for (int i = 0; i < num_tracks; i++) {
            int tid = tracks[i].id;
            if (hyp.track_to_meas.find(tid) == hyp.track_to_meas.end()) {
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
                    hyp.active_labels.insert(tid);
                }
            }
        }

        // 削除済みトラック: 仮説から除去
        auto it = hyp.track_to_meas.begin();
        while (it != hyp.track_to_meas.end()) {
            if (current_track_ids.find(it->first) == current_track_ids.end()) {
                hyp.track_to_meas_pos.erase(it->first);
                hyp.active_labels.erase(it->first);
                it = hyp.track_to_meas.erase(it);
            } else {
                ++it;
            }
        }
    }

    // === Phase 6: スコア加重マージナル確率 + 存在確率 + カーディナリティ ===
    if (!hypotheses_.empty()) {
        // 6a. 仮説スコアを正規化重みに変換（ソフトマックス）
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

        // 6b. 各トラックのマージナル確率計算
        for (int i = 0; i < num_tracks; i++) {
            int tid = tracks[i].id;
            auto it0 = hypotheses_[0].track_to_meas.find(tid);
            int best_meas = (it0 != hypotheses_[0].track_to_meas.end()) ? it0->second : -1;

            float miss_weight = 0.0f;
            float agree_weight = 0.0f;
            float label_active_weight = 0.0f;  // GLMB: ラベルがアクティブな仮説の重み

            // マージナル観測確率（どの観測が最も確率が高いか）
            std::unordered_map<int, float> meas_marginal;

            for (size_t h = 0; h < hypotheses_.size(); h++) {
                auto hit = hypotheses_[h].track_to_meas.find(tid);
                int hyp_meas = (hit != hypotheses_[h].track_to_meas.end()) ? hit->second : -1;

                if (hyp_meas < 0 || hyp_meas >= num_meas) {
                    miss_weight += hyp_weights[h];
                } else {
                    meas_marginal[hyp_meas] += hyp_weights[h];
                }

                if (hyp_meas == best_meas) {
                    agree_weight += hyp_weights[h];
                }

                // GLMB: このラベルがアクティブな仮説の重み合計
                if (hypotheses_[h].active_labels.count(tid) > 0) {
                    label_active_weight += hyp_weights[h];
                }
            }

            // ベスト観測のマージナル確率
            float best_meas_prob = 0.0f;
            int best_marginal_meas = -1;
            for (const auto& [meas_idx, prob] : meas_marginal) {
                if (prob > best_meas_prob) {
                    best_meas_prob = prob;
                    best_marginal_meas = meas_idx;
                }
            }

            result.track_updates[i].best_meas_index = best_meas;
            result.track_updates[i].best_meas_prob = best_meas_prob;
            result.track_updates[i].assignment_confidence = agree_weight;
            result.track_updates[i].miss_prob = miss_weight;

            // 6c. ベイジアン存在確率更新（GLMB改良版）
            float r_i = tracks[i].existence_prob;
            float denom = 1.0f - r_i * Pd;
            if (denom < 1e-10f) denom = 1e-10f;
            float r_miss = r_i * (1.0f - Pd) / denom;

            // 検出時の存在確率: 1.0固定ではなく、ベスト観測の品質に基づく
            // 近い検出(maha<3σ)→高確信、遠い検出(maha>10σ)→低確信
            // これによりクラッタからの偽検出で存在確率が急上昇するのを防ぐ
            float r_detect = r_miss;  // default: no good detection
            int best_j = result.track_updates[i].best_meas_index;
            if (best_j >= 0 && best_j < num_meas) {
                float maha_sq_best = maha_sq_matrix[i][best_j];
                // Gaussian-like quality decay: σ_q^2 = 18 → at 3σ: 0.61, 6σ: 0.14, 10σ: 0.004
                float quality = std::exp(-maha_sq_best / 36.0f);
                r_detect = r_miss + (1.0f - r_miss) * quality;
                // TENTATIVE tracks: further discount to suppress clutter
                if (tracks[i].track_state == TrackState::TENTATIVE) {
                    r_detect = r_miss + (r_detect - r_miss) * 0.7f;
                }
            }
            float r_new = miss_weight * r_miss + (1.0f - miss_weight) * r_detect;

            // GLMB: label activity as soft modulation (not hard penalty)
            if (label_active_weight < 0.3f) {
                r_new *= 0.5f + 0.5f * label_active_weight / 0.3f;
            }

            // Apply survival probability only on miss (not every frame)
            float effective_Ps = miss_weight * Ps + (1.0f - miss_weight) * 1.0f;
            r_new *= effective_Ps;

            r_new = std::max(0.001f, std::min(0.999f, r_new));

            result.track_updates[i].existence_prob = r_new;
        }

        // 6d. カーディナリティ分布計算（GLMB固有）
        result.cardinality_distribution = computeCardinalityDistribution(
            hypotheses_, hyp_weights, num_tracks);

        // E[N] = Σ n * p(N=n)
        float expected_n = 0.0f;
        for (int n = 0; n <= num_tracks; n++) {
            if (n < static_cast<int>(result.cardinality_distribution.size())) {
                expected_n += n * result.cardinality_distribution[n];
            }
        }
        result.estimated_num_targets = expected_n;
    }

    // === Phase 7: ベスト仮説 + 品質ガード → base_result ===
    if (!hypotheses_.empty()) {
        const auto& best_hyp = hypotheses_[0];

        for (int i = 0; i < num_tracks; i++) {
            int tid = tracks[i].id;
            auto it = best_hyp.track_to_meas.find(tid);
            if (it != best_hyp.track_to_meas.end() && it->second >= 0 && it->second < num_meas) {
                int meas_idx = it->second;
                // Quality guard: reject assignments with very high maha distance
                // Tighter guard (5σ) ensures only high-quality measurements update tracks,
                // reducing position RMSE and OSPA
                float maha_best = maha_sq_matrix[i][meas_idx];
                if (maha_best < 25.0f) {  // < 5σ
                    result.base_result.track_to_meas[i] = meas_idx;
                    result.base_result.meas_to_track[meas_idx] = i;
                    result.track_updates[i].best_meas_index = meas_idx;
                }
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
