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
    track.covariance = getInitialCovariance();
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

    // レーダー観測を直交座標に変換
    float r = meas.range;
    float az = meas.azimuth;

    float x = r * std::cos(az);
    float y = r * std::sin(az);

    // 速度はドップラーから推定（視線方向のみ）
    float vx = meas.doppler * std::cos(az);
    float vy = meas.doppler * std::sin(az);

    // 加速度は初期値0
    float ax = 0.0f;
    float ay = 0.0f;

    state << x, y, vx, vy, ax, ay;
    return state;
}

StateCov TrackManager::getInitialCovariance() const {
    StateCov cov;
    cov.setZero();

    // 位置の不確かさ（レーダー精度に基づく）
    cov(0, 0) = 100.0f * 100.0f;  // x: 100m std
    cov(1, 1) = 100.0f * 100.0f;  // y: 100m std

    // 速度の不確かさ（大きめに設定）
    cov(2, 2) = 50.0f * 50.0f;    // vx: 50 m/s std
    cov(3, 3) = 50.0f * 50.0f;    // vy: 50 m/s std

    // 加速度の不確かさ
    cov(4, 4) = 10.0f * 10.0f;    // ax: 10 m/s² std
    cov(5, 5) = 10.0f * 10.0f;    // ay: 10 m/s² std

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
