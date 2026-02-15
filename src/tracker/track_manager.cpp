#include "tracker/track_manager.hpp"
#include <iostream>
#include <cmath>

namespace fasttracker {

TrackManager::TrackManager(const AssociationParams& params)
    : params_(params),
      next_track_id_(0),
      total_tracks_created_(0),
      total_tracks_confirmed_(0),
      total_tracks_deleted_(0)
{
}

int TrackManager::initializeTrack(const Measurement& measurement) {
    Track track;
    track.id = next_track_id_++;
    track.state = measurementToState(measurement);
    track.covariance = getInitialCovariance(measurement.range);
    track.track_state = TrackState::TENTATIVE;
    track.hits = 1;
    track.misses = 0;
    track.last_update_time = measurement.timestamp;

    tracks_[track.id] = track;
    total_tracks_created_++;

    return track.id;
}

void TrackManager::updateTrack(int track_id, const StateVector& state,
                               const StateCov& covariance, double time) {
    if (!hasTrack(track_id)) {
        throw std::runtime_error("Track ID not found");
    }

    Track& track = tracks_[track_id];
    track.state = state;
    track.covariance = covariance;
    track.last_update_time = time;

    updateTrackState(track, time, true);
}

void TrackManager::predictOnlyTrack(int track_id, const StateVector& state,
                                    const StateCov& covariance, double time) {
    if (!hasTrack(track_id)) {
        throw std::runtime_error("Track ID not found");
    }

    Track& track = tracks_[track_id];
    track.state = state;
    track.covariance = covariance;
    track.last_update_time = time;

    updateTrackState(track, time, false);
}

void TrackManager::updateTrackState(Track& track, double time, bool has_measurement) {
    if (has_measurement) {
        track.hits++;
        track.misses = 0;

        // 仮トラックを確定に昇格
        if (track.track_state == TrackState::TENTATIVE) {
            // N out of M ルール: M回のうちN回観測されたら確定
            if (track.hits >= params_.confirm_hits) {
                track.track_state = TrackState::CONFIRMED;
                total_tracks_confirmed_++;
            }
        } else if (track.track_state == TrackState::LOST) {
            // 消失トラックが再発見された場合
            track.track_state = TrackState::CONFIRMED;
        }
    } else {
        track.misses++;

        // 消失判定
        if (track.misses >= params_.delete_misses) {
            track.track_state = TrackState::LOST;
        }
    }
}

void TrackManager::pruneLostTracks() {
    auto it = tracks_.begin();
    while (it != tracks_.end()) {
        if (it->second.track_state == TrackState::LOST) {
            total_tracks_deleted_++;
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<Track> TrackManager::getConfirmedTracks() const {
    std::vector<Track> confirmed;
    for (const auto& pair : tracks_) {
        if (pair.second.track_state == TrackState::CONFIRMED) {
            confirmed.push_back(pair.second);
        }
    }
    return confirmed;
}

std::vector<Track> TrackManager::getAllTracks() const {
    std::vector<Track> all_tracks;
    for (const auto& pair : tracks_) {
        if (pair.second.track_state != TrackState::LOST) {
            all_tracks.push_back(pair.second);
        }
    }
    return all_tracks;
}

const Track& TrackManager::getTrack(int track_id) const {
    auto it = tracks_.find(track_id);
    if (it == tracks_.end()) {
        throw std::runtime_error("Track ID not found");
    }
    return it->second;
}

Track& TrackManager::getTrackMutable(int track_id) {
    auto it = tracks_.find(track_id);
    if (it == tracks_.end()) {
        throw std::runtime_error("Track ID not found");
    }
    return it->second;
}

bool TrackManager::hasTrack(int track_id) const {
    return tracks_.find(track_id) != tracks_.end();
}

int TrackManager::getNumConfirmedTracks() const {
    int count = 0;
    for (const auto& pair : tracks_) {
        if (pair.second.track_state == TrackState::CONFIRMED) {
            count++;
        }
    }
    return count;
}

StateVector TrackManager::measurementToState(const Measurement& meas) const {
    StateVector state;

    // レーダー観測を直交座標に変換（センサー位置オフセットを加算）
    float r = meas.range;
    float az = meas.azimuth;

    float x = r * std::cos(az) + sensor_x_;
    float y = r * std::sin(az) + sensor_y_;

    // 速度はドップラーから推定（視線方向のみ）
    float vx = meas.doppler * std::cos(az);
    float vy = meas.doppler * std::sin(az);

    // 加速度は初期値0
    float ax = 0.0f;
    float ay = 0.0f;

    state << x, y, vx, vy, ax, ay;
    return state;
}

StateCov TrackManager::getInitialCovariance(float range) const {
    StateCov cov;
    cov.setZero();

    // レンジに基づく位置不確かさのスケーリング
    float azimuth_noise = meas_noise_.azimuth_noise;
    float range_noise = meas_noise_.range_noise;
    float pos_std = std::sqrt(range_noise * range_noise +
                              (range * azimuth_noise) * (range * azimuth_noise));
    pos_std = std::max(pos_std, 100.0f);  // 最低100m

    // 速度: ドップラーは視線方向のみなので横方向速度は不明
    // レンジが大きいほど速度不確かさも大きい
    float vel_std = std::max(50.0f, range * 0.001f);  // レンジの0.1% or 最低50m/s

    // 加速度: 弾道ミサイルでは大きな加速度が期待される
    float accel_std = 30.0f;

    cov(0, 0) = pos_std * pos_std;
    cov(1, 1) = pos_std * pos_std;
    cov(2, 2) = vel_std * vel_std;
    cov(3, 3) = vel_std * vel_std;
    cov(4, 4) = accel_std * accel_std;
    cov(5, 5) = accel_std * accel_std;

    return cov;
}

void TrackManager::resetStatistics() {
    total_tracks_created_ = 0;
    total_tracks_confirmed_ = 0;
    total_tracks_deleted_ = 0;
}

void TrackManager::printStatistics() const {
    std::cout << "=== Track Manager Statistics ===" << std::endl;
    std::cout << "Active tracks: " << getNumTracks() << std::endl;
    std::cout << "Confirmed tracks: " << getNumConfirmedTracks() << std::endl;
    std::cout << "Total created: " << total_tracks_created_ << std::endl;
    std::cout << "Total confirmed: " << total_tracks_confirmed_ << std::endl;
    std::cout << "Total deleted: " << total_tracks_deleted_ << std::endl;
    std::cout << "================================" << std::endl;
}

} // namespace fasttracker
