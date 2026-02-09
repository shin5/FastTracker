#include "evaluation/tracking_evaluator.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <iomanip>
#include <numeric>

namespace fasttracker {

TrackingEvaluator::TrackingEvaluator(float ospa_cutoff, int ospa_order)
    : ospa_cutoff_(ospa_cutoff), ospa_order_(ospa_order)
{
}

TrackingEvaluator::~TrackingEvaluator() {
}

void TrackingEvaluator::update(const std::vector<Track>& tracks,
                               const std::vector<StateVector>& ground_truth,
                               int num_measurements,
                               double timestamp)
{
    FrameResult result;
    result.timestamp = timestamp;
    result.num_ground_truth = static_cast<int>(ground_truth.size());
    result.num_tracks = static_cast<int>(tracks.size());
    result.num_measurements = num_measurements;

    // 確定トラック数をカウント
    for (const auto& track : tracks) {
        if (track.track_state == TrackState::CONFIRMED) {
            result.num_confirmed_tracks++;
        }
    }

    // トラックと真値の割り当て
    std::vector<int> track_to_truth;
    std::vector<int> truth_to_track;
    assignTracksToTruth(tracks, ground_truth, track_to_truth, truth_to_track);

    // 位置・速度誤差の計算
    float sum_pos_error = 0.0f;
    float sum_vel_error = 0.0f;
    int num_matched = 0;

    for (size_t i = 0; i < tracks.size(); i++) {
        if (track_to_truth[i] >= 0) {
            float pos_err = computePositionError(tracks[i].state,
                                                ground_truth[track_to_truth[i]]);
            float vel_err = computeVelocityError(tracks[i].state,
                                                ground_truth[track_to_truth[i]]);
            sum_pos_error += pos_err;
            sum_vel_error += vel_err;
            num_matched++;
        }
    }

    if (num_matched > 0) {
        result.avg_position_error = sum_pos_error / num_matched;
        result.avg_velocity_error = sum_vel_error / num_matched;
    }

    // OSPA距離
    result.ospa_distance = computeOSPA(tracks, ground_truth);

    // 検出統計
    result.true_positives = num_matched;
    result.false_positives = static_cast<int>(tracks.size()) - num_matched;
    result.false_negatives = static_cast<int>(ground_truth.size()) - num_matched;

    // トラック履歴更新
    updateTrackHistories(tracks, track_to_truth, timestamp);

    history_.push_back(result);
}

void TrackingEvaluator::assignTracksToTruth(
    const std::vector<Track>& tracks,
    const std::vector<StateVector>& ground_truth,
    std::vector<int>& track_to_truth,
    std::vector<int>& truth_to_track) const
{
    int n_tracks = static_cast<int>(tracks.size());
    int n_truth = static_cast<int>(ground_truth.size());

    track_to_truth.assign(n_tracks, -1);
    truth_to_track.assign(n_truth, -1);

    if (n_tracks == 0 || n_truth == 0) return;

    // コスト行列計算（位置ベース）
    std::vector<std::vector<float>> cost_matrix(n_tracks,
                                                std::vector<float>(n_truth));

    for (int i = 0; i < n_tracks; i++) {
        for (int j = 0; j < n_truth; j++) {
            cost_matrix[i][j] = computeDistance(tracks[i].state, ground_truth[j]);
        }
    }

    // 貪欲法で割り当て（簡易版）
    std::vector<bool> track_assigned(n_tracks, false);
    std::vector<bool> truth_assigned(n_truth, false);
    float threshold = 50.0f;  // 50m以内なら同一目標とみなす

    // 距離が近い順にマッチング
    for (int iter = 0; iter < std::min(n_tracks, n_truth); iter++) {
        float min_cost = std::numeric_limits<float>::max();
        int best_track = -1;
        int best_truth = -1;

        for (int i = 0; i < n_tracks; i++) {
            if (track_assigned[i]) continue;
            for (int j = 0; j < n_truth; j++) {
                if (truth_assigned[j]) continue;
                if (cost_matrix[i][j] < min_cost) {
                    min_cost = cost_matrix[i][j];
                    best_track = i;
                    best_truth = j;
                }
            }
        }

        if (best_track >= 0 && min_cost < threshold) {
            track_to_truth[best_track] = best_truth;
            truth_to_track[best_truth] = best_track;
            track_assigned[best_track] = true;
            truth_assigned[best_truth] = true;
        } else {
            break;
        }
    }
}

float TrackingEvaluator::computeOSPA(const std::vector<Track>& tracks,
                                    const std::vector<StateVector>& ground_truth) const
{
    int m = static_cast<int>(tracks.size());
    int n = static_cast<int>(ground_truth.size());

    if (m == 0 && n == 0) return 0.0f;
    if (m == 0 || n == 0) {
        return ospa_cutoff_;
    }

    // OSPA距離の計算（簡易版）
    // 完全な実装は最適割り当て問題の解が必要

    std::vector<int> track_to_truth;
    std::vector<int> truth_to_track;
    assignTracksToTruth(tracks, ground_truth, track_to_truth, truth_to_track);

    float sum_dist = 0.0f;
    int num_matched = 0;

    for (size_t i = 0; i < tracks.size(); i++) {
        if (track_to_truth[i] >= 0) {
            float dist = computeDistance(tracks[i].state,
                                        ground_truth[track_to_truth[i]]);
            dist = std::min(dist, ospa_cutoff_);
            sum_dist += std::pow(dist, ospa_order_);
            num_matched++;
        }
    }

    // カーディナリティペナルティ
    int cardinality_diff = std::abs(m - n);
    sum_dist += cardinality_diff * std::pow(ospa_cutoff_, ospa_order_);

    int max_count = std::max(m, n);
    float ospa = std::pow(sum_dist / max_count, 1.0f / ospa_order_);

    return ospa;
}

AccuracyMetrics TrackingEvaluator::computeAccuracyMetrics() const {
    AccuracyMetrics metrics;

    if (history_.empty()) return metrics;

    float sum_pos_sq = 0.0f;
    float sum_pos_abs = 0.0f;
    float sum_vel_sq = 0.0f;
    float sum_vel_abs = 0.0f;
    float sum_ospa = 0.0f;
    int count = 0;

    for (const auto& frame : history_) {
        if (frame.avg_position_error > 0.0f) {
            sum_pos_sq += frame.avg_position_error * frame.avg_position_error;
            sum_pos_abs += frame.avg_position_error;
            sum_vel_sq += frame.avg_velocity_error * frame.avg_velocity_error;
            sum_vel_abs += frame.avg_velocity_error;
            count++;
        }
        sum_ospa += frame.ospa_distance;
    }

    if (count > 0) {
        metrics.position_rmse = std::sqrt(sum_pos_sq / count);
        metrics.position_mae = sum_pos_abs / count;
        metrics.velocity_rmse = std::sqrt(sum_vel_sq / count);
        metrics.velocity_mae = sum_vel_abs / count;
    }

    metrics.ospa_distance = sum_ospa / history_.size();

    return metrics;
}

TrackQualityMetrics TrackingEvaluator::computeTrackQualityMetrics() const {
    TrackQualityMetrics metrics;

    if (track_histories_.empty()) return metrics;

    float total_length = 0.0f;
    int total_tracks = 0;
    int confirmed_tracks = 0;
    int false_tracks = 0;

    for (const auto& pair : track_histories_) {
        const TrackHistory& hist = pair.second;

        float length = hist.end_time - hist.start_time;
        total_length += length;
        total_tracks++;

        if (hist.was_confirmed) {
            confirmed_tracks++;
        }

        if (hist.assigned_truth_id < 0) {
            false_tracks++;
        }

        // トラック継続性分類（簡易版）
        if (hist.num_updates > history_.size() * 0.8) {
            metrics.mostly_tracked++;
        } else if (hist.num_updates > history_.size() * 0.2) {
            metrics.partially_tracked++;
        } else {
            metrics.mostly_lost++;
        }
    }

    if (total_tracks > 0) {
        metrics.average_track_length = total_length / total_tracks;
        metrics.confirmation_rate = static_cast<float>(confirmed_tracks) / total_tracks;
        metrics.false_track_rate = static_cast<float>(false_tracks) / total_tracks;
    }

    // トラック純度（簡易版：1 - 偽トラック率）
    metrics.track_purity = 1.0f - metrics.false_track_rate;

    return metrics;
}

DetectionMetrics TrackingEvaluator::computeDetectionMetrics() const {
    DetectionMetrics metrics;

    for (const auto& frame : history_) {
        metrics.true_positives += frame.true_positives;
        metrics.false_positives += frame.false_positives;
        metrics.false_negatives += frame.false_negatives;
    }

    int total_detections = metrics.true_positives + metrics.false_positives;
    int total_truth = metrics.true_positives + metrics.false_negatives;

    if (total_detections > 0) {
        metrics.precision = static_cast<float>(metrics.true_positives) / total_detections;
    }

    if (total_truth > 0) {
        metrics.recall = static_cast<float>(metrics.true_positives) / total_truth;
    }

    if (metrics.precision + metrics.recall > 0) {
        metrics.f1_score = 2.0f * metrics.precision * metrics.recall /
                          (metrics.precision + metrics.recall);
    }

    // 関連付け精度
    if (total_detections > 0) {
        metrics.association_accuracy = metrics.precision;
    }

    return metrics;
}

float TrackingEvaluator::computePositionError(const StateVector& state1,
                                              const StateVector& state2) const
{
    float dx = state1(0) - state2(0);
    float dy = state1(1) - state2(1);
    return std::sqrt(dx * dx + dy * dy);
}

float TrackingEvaluator::computeVelocityError(const StateVector& state1,
                                              const StateVector& state2) const
{
    float dvx = state1(2) - state2(2);
    float dvy = state1(3) - state2(3);
    return std::sqrt(dvx * dvx + dvy * dvy);
}

float TrackingEvaluator::computeDistance(const StateVector& state1,
                                        const StateVector& state2) const
{
    return computePositionError(state1, state2);
}

void TrackingEvaluator::updateTrackHistories(
    const std::vector<Track>& tracks,
    const std::vector<int>& track_to_truth,
    double timestamp)
{
    for (size_t i = 0; i < tracks.size(); i++) {
        int track_id = tracks[i].id;

        if (track_histories_.find(track_id) == track_histories_.end()) {
            // 新規トラック
            TrackHistory hist;
            hist.track_id = track_id;
            hist.start_time = timestamp;
            hist.end_time = timestamp;
            hist.num_updates = 1;
            hist.was_confirmed = (tracks[i].track_state == TrackState::CONFIRMED);
            hist.assigned_truth_id = track_to_truth[i];
            track_histories_[track_id] = hist;
        } else {
            // 既存トラック更新
            TrackHistory& hist = track_histories_[track_id];
            hist.end_time = timestamp;
            hist.num_updates++;
            if (tracks[i].track_state == TrackState::CONFIRMED) {
                hist.was_confirmed = true;
            }
            if (track_to_truth[i] >= 0) {
                hist.assigned_truth_id = track_to_truth[i];
            }
        }
    }
}

void TrackingEvaluator::printSummary() const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "=== Tracking Evaluation Summary ===" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[データ統計]" << std::endl;
    std::cout << "  総フレーム数: " << history_.size() << std::endl;

    if (!history_.empty()) {
        int total_truth = 0;
        int total_tracks = 0;
        for (const auto& frame : history_) {
            total_truth += frame.num_ground_truth;
            total_tracks += frame.num_tracks;
        }
        std::cout << "  平均真値目標数: " << (total_truth / (int)history_.size()) << std::endl;
        std::cout << "  平均トラック数: " << (total_tracks / (int)history_.size()) << std::endl;
    }

    // 精度メトリクス
    auto acc_metrics = computeAccuracyMetrics();
    std::cout << "\n[精度メトリクス]" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  位置RMSE: " << acc_metrics.position_rmse << " m" << std::endl;
    std::cout << "  位置MAE:  " << acc_metrics.position_mae << " m" << std::endl;
    std::cout << "  速度RMSE: " << acc_metrics.velocity_rmse << " m/s" << std::endl;
    std::cout << "  速度MAE:  " << acc_metrics.velocity_mae << " m/s" << std::endl;
    std::cout << "  平均OSPA距離: " << acc_metrics.ospa_distance << " m" << std::endl;

    // 検出メトリクス
    auto det_metrics = computeDetectionMetrics();
    std::cout << "\n[検出メトリクス]" << std::endl;
    std::cout << "  True Positives:  " << det_metrics.true_positives << std::endl;
    std::cout << "  False Positives: " << det_metrics.false_positives << std::endl;
    std::cout << "  False Negatives: " << det_metrics.false_negatives << std::endl;
    std::cout << std::setprecision(4);
    std::cout << "  適合率（Precision）: " << (det_metrics.precision * 100.0f) << " %" << std::endl;
    std::cout << "  再現率（Recall）:    " << (det_metrics.recall * 100.0f) << " %" << std::endl;
    std::cout << "  F1スコア:            " << (det_metrics.f1_score * 100.0f) << " %" << std::endl;

    // トラック品質メトリクス
    auto quality_metrics = computeTrackQualityMetrics();
    std::cout << "\n[トラック品質]" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "  平均トラック継続時間: " << quality_metrics.average_track_length << " s" << std::endl;
    std::cout << "  トラック確定率: " << (quality_metrics.confirmation_rate * 100.0f) << " %" << std::endl;
    std::cout << "  偽トラック率:   " << (quality_metrics.false_track_rate * 100.0f) << " %" << std::endl;
    std::cout << "  トラック純度:   " << (quality_metrics.track_purity * 100.0f) << " %" << std::endl;
    std::cout << "\n  ほぼ全期間追尾（MT）: " << quality_metrics.mostly_tracked << std::endl;
    std::cout << "  部分的に追尾（PT）:   " << quality_metrics.partially_tracked << std::endl;
    std::cout << "  ほとんど失敗（ML）:   " << quality_metrics.mostly_lost << std::endl;

    std::cout << "\n========================================" << std::endl;
}

void TrackingEvaluator::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // ヘッダー
    file << "timestamp,num_ground_truth,num_tracks,num_confirmed,num_measurements,"
         << "avg_position_error,avg_velocity_error,ospa_distance,"
         << "true_positives,false_positives,false_negatives\n";

    // データ
    for (const auto& frame : history_) {
        file << frame.timestamp << ","
             << frame.num_ground_truth << ","
             << frame.num_tracks << ","
             << frame.num_confirmed_tracks << ","
             << frame.num_measurements << ","
             << frame.avg_position_error << ","
             << frame.avg_velocity_error << ","
             << frame.ospa_distance << ","
             << frame.true_positives << ","
             << frame.false_positives << ","
             << frame.false_negatives << "\n";
    }

    file.close();
    std::cout << "評価結果をエクスポート: " << filename << std::endl;
}

void TrackingEvaluator::reset() {
    history_.clear();
    track_histories_.clear();
}

} // namespace fasttracker
