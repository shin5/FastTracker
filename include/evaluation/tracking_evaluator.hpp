#ifndef FASTTRACKER_TRACKING_EVALUATOR_HPP
#define FASTTRACKER_TRACKING_EVALUATOR_HPP

#include "utils/types.hpp"
#include <vector>
#include <map>
#include <string>
#include <fstream>

namespace fasttracker {

// 精度メトリクス
struct AccuracyMetrics {
    float position_rmse;        // 位置RMSE [m]
    float position_mae;         // 位置平均絶対誤差 [m]
    float velocity_rmse;        // 速度RMSE [m/s]
    float velocity_mae;         // 速度平均絶対誤差 [m/s]
    float ospa_distance;        // OSPA距離

    AccuracyMetrics() : position_rmse(0.0f), position_mae(0.0f),
                        velocity_rmse(0.0f), velocity_mae(0.0f),
                        ospa_distance(0.0f) {}
};

// トラック品質メトリクス
struct TrackQualityMetrics {
    float average_track_length;      // 平均トラック継続時間 [s]
    float track_fragmentation_rate;  // トラック分断率
    int num_id_switches;             // IDスイッチ回数
    float track_purity;              // トラック純度
    float confirmation_rate;         // トラック確定率
    float false_track_rate;          // 偽トラック率

    int mostly_tracked;              // ほぼ全期間追尾
    int partially_tracked;           // 部分的に追尾
    int mostly_lost;                 // ほとんど追尾失敗

    TrackQualityMetrics() : average_track_length(0.0f),
                           track_fragmentation_rate(0.0f),
                           num_id_switches(0),
                           track_purity(0.0f),
                           confirmation_rate(0.0f),
                           false_track_rate(0.0f),
                           mostly_tracked(0),
                           partially_tracked(0),
                           mostly_lost(0) {}
};

// 検出メトリクス
struct DetectionMetrics {
    int true_positives;          // 正検出数
    int false_positives;         // 誤検出数
    int false_negatives;         // 未検出数

    float precision;             // 適合率
    float recall;                // 再現率
    float f1_score;              // F1値
    float association_accuracy;  // 関連付け精度

    DetectionMetrics() : true_positives(0), false_positives(0),
                        false_negatives(0), precision(0.0f),
                        recall(0.0f), f1_score(0.0f),
                        association_accuracy(0.0f) {}
};

// フレーム結果
struct FrameResult {
    double timestamp;
    int num_ground_truth;        // 真の目標数
    int num_tracks;              // トラック数
    int num_confirmed_tracks;    // 確定トラック数
    int num_measurements;        // 観測数

    float avg_position_error;    // 平均位置誤差
    float avg_velocity_error;    // 平均速度誤差
    float ospa_distance;         // OSPA距離

    int true_positives;          // この時刻の正検出数
    int false_positives;         // この時刻の誤検出数
    int false_negatives;         // この時刻の未検出数

    FrameResult() : timestamp(0.0), num_ground_truth(0), num_tracks(0),
                   num_confirmed_tracks(0), num_measurements(0),
                   avg_position_error(0.0f), avg_velocity_error(0.0f),
                   ospa_distance(0.0f), true_positives(0),
                   false_positives(0), false_negatives(0) {}
};

// トラック履歴
struct TrackHistory {
    int track_id;
    double start_time;
    double end_time;
    int num_updates;
    int num_misses;
    bool was_confirmed;
    int assigned_truth_id;  // -1 = 偽トラック

    TrackHistory() : track_id(-1), start_time(0.0), end_time(0.0),
                    num_updates(0), num_misses(0), was_confirmed(false),
                    assigned_truth_id(-1) {}
};

class TrackingEvaluator {
public:
    TrackingEvaluator(float ospa_cutoff = 100.0f, int ospa_order = 2);
    ~TrackingEvaluator();

    // フレーム更新
    void update(const std::vector<Track>& tracks,
                const std::vector<StateVector>& ground_truth,
                int num_measurements,
                double timestamp);

    // メトリクス計算
    AccuracyMetrics computeAccuracyMetrics() const;
    TrackQualityMetrics computeTrackQualityMetrics() const;
    DetectionMetrics computeDetectionMetrics() const;

    // OSPA距離計算
    float computeOSPA(const std::vector<Track>& tracks,
                      const std::vector<StateVector>& ground_truth) const;

    // 統計情報
    void printSummary() const;
    void exportToCSV(const std::string& filename) const;
    void reset();

    // アクセサ
    const std::vector<FrameResult>& getHistory() const { return history_; }
    int getTotalFrames() const { return static_cast<int>(history_.size()); }

private:
    // パラメータ
    float ospa_cutoff_;      // OSPA距離のカットオフ [m]
    int ospa_order_;         // OSPA距離の次数

    // 履歴データ
    std::vector<FrameResult> history_;
    std::map<int, TrackHistory> track_histories_;

    // 内部処理
    void assignTracksToTruth(const std::vector<Track>& tracks,
                            const std::vector<StateVector>& ground_truth,
                            std::vector<int>& track_to_truth,
                            std::vector<int>& truth_to_track) const;

    float computePositionError(const StateVector& state1,
                              const StateVector& state2) const;

    float computeVelocityError(const StateVector& state1,
                              const StateVector& state2) const;

    float computeDistance(const StateVector& state1,
                         const StateVector& state2) const;

    void updateTrackHistories(const std::vector<Track>& tracks,
                             const std::vector<int>& track_to_truth,
                             double timestamp);
};

} // namespace fasttracker

#endif // FASTTRACKER_TRACKING_EVALUATOR_HPP
