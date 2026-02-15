#ifndef FASTTRACKER_TRACK_MANAGER_HPP
#define FASTTRACKER_TRACK_MANAGER_HPP

#include <vector>
#include <map>
#include "utils/types.hpp"

namespace fasttracker {

/**
 * @brief トラック管理クラス
 *
 * トラックのライフサイクル（初期化、確定、消失）を管理します。
 */
class TrackManager {
public:
    /**
     * @brief コンストラクタ
     * @param params データアソシエーションパラメータ
     */
    explicit TrackManager(const AssociationParams& params = AssociationParams());

    /**
     * @brief 新しいトラックを初期化
     * @param measurement 初期観測
     * @return 新しいトラックID
     */
    int initializeTrack(const Measurement& measurement);

    /**
     * @brief トラックを更新（観測が割り当てられた場合）
     * @param track_id トラックID
     * @param state 更新後の状態
     * @param covariance 更新後の共分散
     * @param time 現在時刻
     */
    void updateTrack(int track_id, const StateVector& state,
                     const StateCov& covariance, double time);

    /**
     * @brief トラックを予測のみ更新（観測が割り当てられなかった場合）
     * @param track_id トラックID
     * @param state 予測状態
     * @param covariance 予測共分散
     * @param time 現在時刻
     */
    void predictOnlyTrack(int track_id, const StateVector& state,
                          const StateCov& covariance, double time);

    /**
     * @brief 消失トラックを削除
     */
    void pruneLostTracks();

    /**
     * @brief 確定トラックのリストを取得
     */
    std::vector<Track> getConfirmedTracks() const;

    /**
     * @brief 全トラック（仮+確定）のリストを取得
     */
    std::vector<Track> getAllTracks() const;

    /**
     * @brief トラックIDからトラックを取得
     */
    const Track& getTrack(int track_id) const;

    /**
     * @brief トラックIDからトラックを取得（変更可能）
     */
    Track& getTrackMutable(int track_id);

    /**
     * @brief トラックが存在するかチェック
     */
    bool hasTrack(int track_id) const;

    /**
     * @brief トラック数を取得
     */
    int getNumTracks() const { return static_cast<int>(tracks_.size()); }

    /**
     * @brief 確定トラック数を取得
     */
    int getNumConfirmedTracks() const;

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
    void setSensorPosition(float x, float y) { sensor_x_ = x; sensor_y_ = y; }

    /**
     * @brief 観測ノイズを設定（初期共分散計算用）
     */
    void setMeasurementNoise(const MeasurementNoise& noise) { meas_noise_ = noise; }

private:
    AssociationParams params_;
    std::map<int, Track> tracks_;  // track_id -> Track
    int next_track_id_;

    // 統計情報
    int total_tracks_created_;
    int total_tracks_confirmed_;
    int total_tracks_deleted_;

    float sensor_x_ = 0.0f;
    float sensor_y_ = 0.0f;
    MeasurementNoise meas_noise_;

    /**
     * @brief 観測から初期状態を推定
     */
    StateVector measurementToState(const Measurement& meas) const;

    /**
     * @brief 初期共分散を設定（レンジに基づくスケーリング）
     */
    StateCov getInitialCovariance(float range) const;

    /**
     * @brief トラック状態を更新
     */
    void updateTrackState(Track& track, double time, bool has_measurement);
};

} // namespace fasttracker

#endif // FASTTRACKER_TRACK_MANAGER_HPP
