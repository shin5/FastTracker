#include "tracker/pmbm_association.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

PMBMAssociation::PMBMAssociation(const AssociationParams& params,
                                   const MeasurementNoise& meas_noise)
    : params_(params), meas_noise_(meas_noise)
{
}

MeasVector PMBMAssociation::predictMeasurement(const StateVector& state,
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

float PMBMAssociation::computeNormalizedDist(const MeasVector& pred_meas,
                                               const Measurement& meas) const
{
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

// ========================================
// スタンドアロンHungarian法（Munkresアルゴリズム）
// data_association.cu から抽出（.cppで使用可能にするため）
// ========================================
std::vector<int> PMBMAssociation::hungarianSolve(
    const std::vector<float>& cost_matrix, int n, int m)
{
    std::vector<int> assignments(n, -1);
    if (n == 0 || m == 0) return assignments;

    const int N = std::max(n, m);

    // パディング済み正方コスト行列を構築
    std::vector<float> C(N * N, 0.0f);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i * N + j] = cost_matrix[i * m + j];

    // スター・プライム・カバー
    std::vector<int> star_col(N, -1);
    std::vector<int> star_row(N, -1);
    std::vector<int> prime_col(N, -1);
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

    const float HUNGARIAN_EPS = 1e-9f;

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
        if (covered >= N) break;

        // Step 4-6 ループ
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
                    // Step 6: 未カバー最小値で調整
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
                // Step 5: 増加パス
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

    // 結果抽出
    for (int i = 0; i < n; i++) {
        int j = star_col[i];
        if (j >= 0 && j < m && cost_matrix[i * m + j] < 1e9f) {
            assignments[i] = j;
        }
    }

    return assignments;
}

// ========================================
// Murty's K-best割当アルゴリズム
// ========================================
std::vector<PMBMAssociation::Assignment> PMBMAssociation::murtyKBest(
    const std::vector<float>& cost_matrix,
    int num_tracks, int num_cols, int K)
{
    std::vector<Assignment> results;
    if (num_tracks == 0 || num_cols == 0 || K <= 0) return results;

    const float BIG_COST = 1e6f;

    // ベスト割当を求める
    std::vector<int> best_assign = hungarianSolve(cost_matrix, num_tracks, num_cols);

    // ベスト割当のコスト計算
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

    // Murty's分割: 優先度キュー（最小コスト順）
    // 各ノードは「一部の割当を固定し、一部を禁止した」サブ問題
    auto cmp = [](const MurtyNode& a, const MurtyNode& b) {
        return a.base_cost + 0 > b.base_cost + 0;  // NOTE: ノードのtotal = base + solve
    };

    // 簡易Murty: ベスト割当の各行を順に禁止して次善解を生成
    // 完全なMurtyはパーティション木を構築するが、K=5程度なら以下で十分
    struct QueueEntry {
        std::vector<float> cost_mat;  // 制約付きコスト行列
        float solved_cost;            // Hungarian解のコスト
        std::vector<int> assignment;  // 解の割当

        bool operator>(const QueueEntry& other) const {
            return solved_cost > other.solved_cost;
        }
    };

    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> pq;

    // 初期分割: ベスト割当の各(i, j)ペアを1つずつ禁止
    for (int i = 0; i < num_tracks; i++) {
        int j_forbidden = best_assign[i];
        if (j_forbidden < 0) continue;

        // コスト行列をコピーし、(i, j_forbidden)を禁止
        std::vector<float> new_cost = cost_matrix;
        new_cost[i * num_cols + j_forbidden] = BIG_COST;

        // 先行行の割当を固定
        for (int ii = 0; ii < i; ii++) {
            int j_fixed = best_assign[ii];
            if (j_fixed < 0) continue;
            // 行iiはj_fixed以外を禁止
            for (int jj = 0; jj < num_cols; jj++) {
                if (jj != j_fixed) {
                    new_cost[ii * num_cols + jj] = BIG_COST;
                }
            }
            // 他の行からj_fixedを禁止
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

    // トップK-1を取り出し
    while (!pq.empty() && static_cast<int>(results.size()) < K) {
        QueueEntry top = pq.top();
        pq.pop();

        // 重複チェック
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

        // この解をさらに分割
        for (int i = 0; i < num_tracks; i++) {
            int j_forbidden = top.assignment[i];
            if (j_forbidden < 0) continue;

            std::vector<float> new_cost = top.cost_mat;
            new_cost[i * num_cols + j_forbidden] = BIG_COST;

            // 先行行を固定
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
// PMBMアソシエーション メイン
// ========================================
PMBMResult PMBMAssociation::associate(const std::vector<Track>& tracks,
                                       const std::vector<Measurement>& measurements,
                                       float sensor_x, float sensor_y, float sensor_z)
{
    int num_tracks = static_cast<int>(tracks.size());
    int num_meas = static_cast<int>(measurements.size());

    PMBMResult result;
    result.base_result.track_to_meas.resize(num_tracks, -1);
    result.base_result.meas_to_track.resize(num_meas, -1);
    result.track_updates.resize(num_tracks);

    // 初期化
    for (int i = 0; i < num_tracks; i++) {
        result.track_updates[i].track_index = i;
        result.track_updates[i].has_gated_meas = false;
        result.track_updates[i].existence_prob = tracks[i].existence_prob;
        result.track_updates[i].miss_prob = 1.0f;
        result.track_updates[i].best_meas_index = -1;
        result.track_updates[i].best_meas_prob = 0.0f;
    }

    if (num_tracks == 0 || num_meas == 0) {
        for (int i = 0; i < num_tracks; i++)
            result.base_result.unassigned_tracks.push_back(i);
        for (int j = 0; j < num_meas; j++)
            result.base_result.unassigned_measurements.push_back(j);
        return result;
    }

    float Pd = params_.pmbm_pd;
    float lambda_c = params_.pmbm_clutter_density;
    float Ps = params_.pmbm_survival_prob;
    int K = params_.pmbm_k_best;

    // === Phase 1: 予測観測 + ゲーティング + 距離・尤度計算 ===
    float noise_stds[MEAS_DIM] = {
        meas_noise_.range_noise,
        meas_noise_.azimuth_noise,
        meas_noise_.elevation_noise,
        meas_noise_.doppler_noise
    };

    // 正規化定数 (4次元対角ガウス) — 仮説重み計算用
    float norm_const = 1.0f;
    for (int d = 0; d < MEAS_DIM; d++) {
        norm_const /= (std::sqrt(2.0f * static_cast<float>(M_PI)) * noise_stds[d]);
    }

    std::vector<MeasVector> pred_meas(num_tracks);
    std::vector<std::vector<int>> gated_meas(num_tracks);
    std::vector<bool> meas_in_any_gate(num_meas, false);

    // マハラノビス距離二乗行列 (num_tracks × num_meas) — 割当コスト用
    std::vector<std::vector<float>> maha_sq_matrix(num_tracks, std::vector<float>(num_meas, 0.0f));
    // 対数尤度行列 (num_tracks × num_meas) — 仮説重み用
    std::vector<std::vector<float>> log_likelihood(num_tracks, std::vector<float>(num_meas, -1e30f));

    for (int i = 0; i < num_tracks; i++) {
        pred_meas[i] = predictMeasurement(tracks[i].state, sensor_x, sensor_y, sensor_z);

        // TENTATIVE航跡にはより狭いゲートを適用（クラッタ取得抑制）
        float effective_gate = params_.pmbm_gate;
        if (tracks[i].track_state == TrackState::TENTATIVE) {
            effective_gate = std::min(params_.pmbm_gate, 40.0f);
        }

        for (int j = 0; j < num_meas; j++) {
            float dist = computeNormalizedDist(pred_meas[i], measurements[j]);
            if (dist < effective_gate) {
                gated_meas[i].push_back(j);
                meas_in_any_gate[j] = true;

                // イノベーション計算
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
                // 対数尤度: log(Pd * L_ij / lambda_c)
                float L_ij = norm_const * std::exp(-0.5f * maha_sq);
                float detection_lr = Pd * L_ij / lambda_c;
                if (detection_lr > 1e-30f) {
                    log_likelihood[i][j] = std::log(detection_lr);
                }
            }
        }

        if (!gated_meas[i].empty()) {
            result.track_updates[i].has_gated_meas = true;
        }
    }

    // ゲートに入らなかった観測 → unassigned_measurements
    for (int j = 0; j < num_meas; j++) {
        if (!meas_in_any_gate[j]) {
            result.base_result.unassigned_measurements.push_back(j);
        }
    }

    // === Phase 2: 拡張コスト行列構築 (マハラノビス距離ベース) ===
    // 割当にはマハラノビス距離を使用 (GNNと同等の割当品質)
    // 左 num_meas 列: maha_sq (ゲート外は BIG_COST)
    // 右 num_tracks 列: gate^2 対角 (ミスコスト = ゲート閾値の二乗)

    int num_cols = num_meas + num_tracks;
    const float BIG_COST = 1e6f;
    float gate_sq = params_.pmbm_gate * params_.pmbm_gate;

    std::vector<float> ext_cost(num_tracks * num_cols, BIG_COST);

    for (int i = 0; i < num_tracks; i++) {
        // 検出列: マハラノビス距離二乗
        for (int j : gated_meas[i]) {
            ext_cost[i * num_cols + j] = maha_sq_matrix[i][j];
        }

        // 未検出列（対角）: 存在確率ベースのミスコスト（航跡優先度）
        // CONFIRMED: 最高優先度 → BIG_COST/2 (観測を確実に確保)
        // TENTATIVE: floor=0.6 → miss_cost=1500 (クラッタ取得を抑制しつつ目標獲得を維持)
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
    std::vector<Assignment> hypotheses = murtyKBest(ext_cost, num_tracks, num_cols, K);

    if (hypotheses.empty()) {
        // フォールバック: 全トラック未割当
        for (int i = 0; i < num_tracks; i++)
            result.base_result.unassigned_tracks.push_back(i);
        return result;
    }

    // === Phase 4: 仮説の重み計算 (対数尤度ベース) + マージナル割当確率 ===
    // 割当はマハラノビス距離で決定済み。仮説の重みは対数尤度で計算。
    // w_k = exp(Σ log_likelihood_k), 正規化して π_k
    float log_miss = std::log(std::max(1.0f - Pd, 1e-10f));  // log(1 - Pd)

    std::vector<float> hyp_log_weights(hypotheses.size());
    for (size_t k = 0; k < hypotheses.size(); k++) {
        float total_log_weight = 0.0f;
        for (int i = 0; i < num_tracks; i++) {
            int col = hypotheses[k].track_to_meas[i];
            if (col >= 0 && col < num_meas) {
                total_log_weight += log_likelihood[i][col];
            } else {
                total_log_weight += log_miss;
            }
        }
        hyp_log_weights[k] = total_log_weight;
    }

    // 数値安定性: max log weight を引く
    float max_log_w = hyp_log_weights[0];
    for (size_t k = 1; k < hypotheses.size(); k++) {
        if (hyp_log_weights[k] > max_log_w) max_log_w = hyp_log_weights[k];
    }

    std::vector<float> weights(hypotheses.size());
    float weight_sum = 0.0f;
    for (size_t k = 0; k < hypotheses.size(); k++) {
        weights[k] = std::exp(hyp_log_weights[k] - max_log_w);
        weight_sum += weights[k];
    }
    if (weight_sum < 1e-30f) weight_sum = 1e-30f;
    for (size_t k = 0; k < hypotheses.size(); k++) {
        weights[k] /= weight_sum;
    }

    // マージナル割当確率
    // marginal_meas[i][j] = P(track_i → meas_j)
    // marginal_miss[i] = P(track_i → miss)
    std::vector<float> marginal_miss(num_tracks, 0.0f);
    std::vector<std::vector<float>> marginal_meas(num_tracks, std::vector<float>(num_meas, 0.0f));

    for (size_t k = 0; k < hypotheses.size(); k++) {
        const auto& h = hypotheses[k];
        for (int i = 0; i < num_tracks; i++) {
            int col = h.track_to_meas[i];
            if (col < 0 || col >= num_meas) {
                // 未検出列に割当（or 未割当）
                marginal_miss[i] += weights[k];
            } else {
                marginal_meas[i][col] += weights[k];
            }
        }
    }

    // === Phase 5: 存在確率更新（ベルヌーイ） ===
    for (int i = 0; i < num_tracks; i++) {
        float r_i = tracks[i].existence_prob;

        // P(miss) × ベルヌーイ更新
        // 未検出時: r_miss = r_i * (1 - Pd) / (1 - r_i * Pd)
        // 検出時: r_det = 1 (確実に存在)
        // 期待値: r_new = marginal_miss * r_miss + (1 - marginal_miss) * 1.0

        float r_miss;
        float denom = 1.0f - r_i * Pd;
        if (denom < 1e-10f) denom = 1e-10f;
        r_miss = r_i * (1.0f - Pd) / denom;

        float r_new = marginal_miss[i] * r_miss + (1.0f - marginal_miss[i]) * 1.0f;

        // 生存確率を乗算
        r_new *= Ps;

        // クランプ
        r_new = std::max(0.001f, std::min(0.999f, r_new));

        result.track_updates[i].existence_prob = r_new;
        result.track_updates[i].miss_prob = marginal_miss[i];
    }

    // === Phase 6: Hungarian最適仮説(hypothesis[0])の割当をbase_resultにセット ===
    // hypothesis[0] = Hungarian法のグローバル最適割当（GNNと同等の割当品質）
    // K仮説のマージナル確率は Phase 5 の存在確率更新にのみ使用
    const auto& best_hyp = hypotheses[0];

    for (int i = 0; i < num_tracks; i++) {
        int col = best_hyp.track_to_meas[i];
        if (col >= 0 && col < num_meas) {
            result.base_result.track_to_meas[i] = col;
            result.base_result.meas_to_track[col] = i;
            result.track_updates[i].best_meas_index = col;

            // ベスト観測のマージナル確率
            result.track_updates[i].best_meas_prob = marginal_meas[i][col];
        }
    }

    // 未割当トラック/観測
    for (int i = 0; i < num_tracks; i++) {
        if (result.base_result.track_to_meas[i] < 0) {
            result.base_result.unassigned_tracks.push_back(i);
        }
    }

    // ベスト仮説で割当されなかった観測も unassigned に追加
    for (int j = 0; j < num_meas; j++) {
        if (result.base_result.meas_to_track[j] < 0 && meas_in_any_gate[j]) {
            result.base_result.unassigned_measurements.push_back(j);
        }
    }

    return result;
}

} // namespace fasttracker
