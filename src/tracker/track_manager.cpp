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
    track.age++;  // M-of-N: フレームカウント

    if (has_measurement) {
        track.hits++;
        track.misses = 0;

        // 仮トラックを確定に昇格（M-of-N: N窓内にM回検出で確定）
        if (track.track_state == TrackState::TENTATIVE) {
            if (track.hits >= params_.confirm_hits) {
                // 確定前に重複チェック: 近傍に既存CONFIRMED航跡があれば昇格を抑制
                if (params_.min_init_distance > 0.0f) {
                    float dedup_dist_sq = params_.min_init_distance * params_.min_init_distance;
                    bool has_nearby_confirmed = false;
                    for (const auto& pair : tracks_) {
                        if (pair.second.id == track.id) continue;
                        if (pair.second.track_state == TrackState::CONFIRMED) {
                            float dx = track.state(0) - pair.second.state(0);
                            float dy = track.state(1) - pair.second.state(1);
                            float dz = track.state(2) - pair.second.state(2);
                            if (dx*dx + dy*dy + dz*dz < dedup_dist_sq) {
                                has_nearby_confirmed = true;
                                break;
                            }
                        }
                    }
                    if (has_nearby_confirmed) {
                        // 近傍にCONFIRMED航跡あり → 重複なので削除
                        track.track_state = TrackState::DELETED;
                        return;
                    }
                }
                track.track_state = TrackState::CONFIRMED;
                total_tracks_confirmed_++;
            }
        } else if (track.track_state == TrackState::LOST) {
            // 消失トラックが再発見された場合
            track.track_state = TrackState::CONFIRMED;
        }
    } else {
        track.misses++;

        // TENTATIVE航跡: M-of-N方式（部分ヒット延長付き）
        // 部分的にヒットがある航跡は確認ウィンドウを延長して生存機会を増やす。
        // これにより、レンジ分解能内の未分解目標が分離後に航跡を確立しやすくなる。
        if (track.track_state == TrackState::TENTATIVE) {
            int effective_window = params_.confirm_window;
            if (track.hits > 0 && track.hits < params_.confirm_hits) {
                // 不足ヒット数に比例して延長: hits=1/3 → +2/3*window, hits=2/3 → +1/3*window
                int remaining = params_.confirm_hits - track.hits;
                effective_window += params_.confirm_window * remaining / std::max(params_.confirm_hits, 1);
            }
            if (track.age >= effective_window && track.hits < params_.confirm_hits) {
                track.track_state = TrackState::DELETED;  // TENTATIVE失敗は削除
                return;
            }
        }

        // CONFIRMED航跡の消失判定
        if (track.track_state == TrackState::CONFIRMED && track.misses >= params_.delete_misses) {
            track.track_state = TrackState::LOST;
        }

        // LOST航跡の最終削除: 猶予期間(+50%)で完全削除
        // LOST状態でも近接チェックの対象になるため、適度に保持
        if (track.track_state == TrackState::LOST) {
            int grace_misses = params_.delete_misses + params_.delete_misses / 2;
            if (track.misses >= grace_misses) {
                track.track_state = TrackState::DELETED;
            }
        }
    }
}

void TrackManager::pruneLostTracks() {
    auto it = tracks_.begin();
    while (it != tracks_.end()) {
        if (it->second.track_state == TrackState::DELETED) {
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
        // LOST航跡も含む（再捕捉猶予期間中のデータアソシエーション参加）
        // DELETEDのみ除外
        if (pair.second.track_state != TrackState::DELETED) {
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

    // レーダー観測を直交座標に変換（3D、センサー位置オフセットを加算）
    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    float r = meas.range;
    float az = meas.azimuth;
    float el = meas.elevation;

    float r_horiz = r * std::cos(el);
    float x = r_horiz * std::cos(az) + sensor_x_;
    float y = r_horiz * std::sin(az) + sensor_y_;
    float z = r * std::sin(el) + sensor_z_;

    // 速度はドップラーから推定（視線方向のみ）
    float vx = meas.doppler * std::cos(el) * std::cos(az);
    float vy = meas.doppler * std::cos(el) * std::sin(az);
    float vz = meas.doppler * std::sin(el);

    state << x, y, z, vx, vy, vz, 0.0f, 0.0f, 0.0f;
    return state;
}

StateCov TrackManager::getInitialCovariance(float range) const {
    // 9×9 対角共分散行列
    // 状態: [x, y, z, vx, vy, vz, ax, ay, az]
    StateCov cov;
    cov.setZero();

    // レンジに基づく位置不確かさのスケーリング
    float azimuth_noise = meas_noise_.azimuth_noise;
    float range_noise = meas_noise_.range_noise;
    float pos_std = std::sqrt(range_noise * range_noise +
                              (range * azimuth_noise) * (range * azimuth_noise));
    pos_std = std::max(pos_std, 100.0f);  // 最低100m

    // 速度: ドップラーは視線方向のみなので横方向速度は不明
    float vel_std = std::max(100.0f, range * 0.001f);  // レンジの0.1% or 最低100m/s

    // 加速度: 弾道ミサイルでは大きな加速度が期待される
    float accel_std = 30.0f;

    // 位置 (x, y, z)
    cov(0, 0) = pos_std * pos_std;
    cov(1, 1) = pos_std * pos_std;
    cov(2, 2) = pos_std * pos_std;

    // 速度 (vx, vy, vz)
    cov(3, 3) = vel_std * vel_std;
    cov(4, 4) = vel_std * vel_std;
    cov(5, 5) = vel_std * vel_std;

    // 加速度 (ax, ay, az)
    cov(6, 6) = accel_std * accel_std;
    cov(7, 7) = accel_std * accel_std;
    cov(8, 8) = accel_std * accel_std;

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
