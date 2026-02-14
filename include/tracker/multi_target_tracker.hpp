#ifndef FASTTRACKER_MULTI_TARGET_TRACKER_HPP
#define FASTTRACKER_MULTI_TARGET_TRACKER_HPP

#include <vector>
#include <memory>
#include "utils/types.hpp"
#include "ukf/ukf.cuh"
#include "ukf/imm_ukf.hpp"
#include "ukf/imm_ukf.cuh"
#include "tracker/track_manager.hpp"
#include "tracker/data_association.cuh"

namespace fasttracker {

/**
 * @brief 多目標追尾メインクラス
 *
 * UKF、トラック管理、データアソシエーションを統合した
 * 完全な多目標追尾システムです。
 */
class MultiTargetTracker {
public:
    /**
     * @brief コンストラクタ
     * @param max_targets 最大トラック数
     * @param ukf_params UKFパラメータ
     * @param assoc_params アソシエーションパラメータ
     * @param process_noise プロセスノイズ
     * @param meas_noise 観測ノイズ
     * @param use_imm IMMフィルタを使用するか（デフォルト: true）
     */
    MultiTargetTracker(
        int max_targets = 2000,
        const UKFParams& ukf_params = UKFParams(),
        const AssociationParams& assoc_params = AssociationParams(),
        const ProcessNoise& process_noise = ProcessNoise(),
        const MeasurementNoise& meas_noise = MeasurementNoise(),
        bool use_imm = true
    );

    ~MultiTargetTracker();

    /**
     * @brief 追尾を更新（予測 + データアソシエーション + 更新）
     * @param measurements 観測データ
     * @param current_time 現在時刻 [s]
     */
    void update(const std::vector<Measurement>& measurements, double current_time);

    /**
     * @brief 確定トラックを取得
     */
    std::vector<Track> getConfirmedTracks() const;

    /**
     * @brief 全トラックを取得
     */
    std::vector<Track> getAllTracks() const;

    /**
     * @brief トラック数を取得
     */
    int getNumTracks() const;

    /**
     * @brief 確定トラック数を取得
     */
    int getNumConfirmedTracks() const;

    /**
     * @brief 前回の更新時刻を取得
     */
    double getLastUpdateTime() const { return last_update_time_; }

    /**
     * @brief 統計情報をリセット
     */
    void resetStatistics();

    /**
     * @brief 統計情報を表示
     */
    void printStatistics() const;

    /**
     * @brief センサー位置を設定
     */
    void setSensorPosition(float x, float y) {
        sensor_x_ = x; sensor_y_ = y;
        if (data_association_) data_association_->setSensorPosition(x, y);
        if (track_manager_) track_manager_->setSensorPosition(x, y);
    }

    /**
     * @brief パフォーマンス統計を取得
     */
    struct PerformanceStats {
        double predict_time_ms;
        double association_time_ms;
        double update_time_ms;
        double total_time_ms;
        int num_tracks;
        int num_measurements;
    };

    const PerformanceStats& getLastPerformanceStats() const {
        return last_perf_stats_;
    }

private:
    // コンポーネント
    std::unique_ptr<UKF> ukf_;
    std::unique_ptr<IMMFilterGPU> imm_gpu_;  // GPU版IMM
    std::unique_ptr<IMMFilter> imm_cpu_;     // CPU版IMM
    std::unique_ptr<TrackManager> track_manager_;
    std::unique_ptr<DataAssociation> data_association_;

    // パラメータ
    int max_targets_;
    UKFParams ukf_params_;
    AssociationParams assoc_params_;
    ProcessNoise process_noise_;
    MeasurementNoise meas_noise_;

    // 状態
    double last_update_time_;
    bool first_update_;
    bool use_imm_;  // IMMフィルタを使用するか
    int imm_gpu_threshold_;  // GPU IMM使用の閾値トラック数（デフォルト: 200）
    float sensor_x_;  // センサーX座標 [m]
    float sensor_y_;  // センサーY座標 [m]

    // パフォーマンス統計
    PerformanceStats last_perf_stats_;

    // 統計情報
    int total_updates_;
    double total_processing_time_;
    int total_measurements_processed_;

    /**
     * @brief 予測ステップ
     * @param dt 時間差分 [s]
     */
    void predictTracks(double dt);

    /**
     * @brief 新規トラックを初期化
     * @param unassigned_measurements 未割り当て観測のインデックス
     * @param measurements 観測データ
     */
    void initializeNewTracks(const std::vector<int>& unassigned_measurements,
                            const std::vector<Measurement>& measurements);

    /**
     * @brief 消失トラックを削除
     */
    void pruneTracks();
};

} // namespace fasttracker

#endif // FASTTRACKER_MULTI_TARGET_TRACKER_HPP
