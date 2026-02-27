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
    track.snr_sum = measurement.snr;
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

        // 仮トラックを確定に昇格
        // 動的confirm_hits: CONFIRMED < 4は2ヒット（高速獲得）、>=4はparams_.confirm_hits（厳格）
        // 健全CONFIRMEDのみカウント: misses>=3の瀕死航跡は後継TENTATIVEの確認を
        // ブロックしないよう除外する。瀕死航跡を含めるとnum_known過大→confirm_hits=4+dedup=5km
        // で後継が確認できず、目標が10+フレーム未追尾になる。
        if (track.track_state == TrackState::TENTATIVE) {
            int num_known = 0;
            for (const auto& pair : tracks_) {
                if (pair.second.track_state == TrackState::CONFIRMED &&
                    pair.second.misses < 3) num_known++;
            }
            // 動的confirm_hits:
            //   num_known=0: 2ヒット（最初の目標を高速獲得）
            //   num_known=1-3: 3ヒット（重複確認を抑制）
            //   num_known>=4: params_.confirm_hits（厳格）
            int effective_confirm_hits;
            if (num_known >= 4) {
                effective_confirm_hits = params_.confirm_hits;
            } else if (num_known >= 1) {
                effective_confirm_hits = std::min(params_.confirm_hits, 3);
            } else {
                effective_confirm_hits = std::min(params_.confirm_hits, 2);
            }
            if (track.hits >= effective_confirm_hits) {
                // SNR品質チェック: 平均SNRが閾値未満のTENTATIVEは確認を拒否
                // 初期化閾値+5dB (=15dB) で確認品質を保証
                float avg_snr = (track.hits > 0) ? track.snr_sum / track.hits : 0.0f;
                float snr_confirm_threshold = params_.min_snr_for_init + 5.0f;
                if (avg_snr < snr_confirm_threshold) {
                    track.track_state = TrackState::DELETED;
                    return;
                }

                // GLMB存在確率ゲート: GLMBのグローバル仮説で低存在確率のTENTATIVEは
                // クラッタ由来の可能性が高い。hits数だけでなく存在確率も確認条件に追加。
                // HGVシナリオ(3目標)では3目標確認後の偽トラック確認を防止。
                // num_known<3（追尾立ち上げ期）は緩和して高速獲得を妨げない。
                // 閾値0.25: 初期存在確率(0.2)よりわずかに高く、1回の良質検出で突破可能。
                if (num_known >= 3 && track.existence_prob < 0.25f) {
                    // 存在確率が閾値未満→まだ確認しない（証拠蓄積を待つ）
                    // ただし confirm_window を超えたら削除
                    if (track.age >= params_.confirm_window * 2) {
                        track.track_state = TrackState::DELETED;
                    }
                    return;
                }

                // 確定前に重複チェック: 既存CONFIRMED航跡との距離チェック
                // LOST航跡は対象外: getAllTracksから除外されており観測・ビームを消費しない。
                // LOSTをチェックするとChange Aで削除された航跡が後継TENTATIVEの確認を
                // ブロックし、トラックスロットが10+フレーム空白になる問題が発生する。
                // 動的dedup: num_known<4は1km（SNR高品質なら200mに緩和）、
                //           num_known>=4は5km（偽トラック増殖防止、既存追尾充足時）
                if (params_.min_init_distance > 0.0f) {
                    float base_dedup = 1000.0f;
                    // 高SNR TENTATIVE（実目標候補）は近接確認を緩和（分離直後でも早期確認）
                    if (track.hits > 0) {
                        float avg_snr_db = track.snr_sum / track.hits;
                        if (avg_snr_db >= 20.0f) {
                            base_dedup = 100.0f;  // 高SNR: 100m（分離後~1.5秒で超過）
                        }
                    }
                    float dedup_dist = (num_known >= 4) ? 5000.0f : base_dedup;
                    float dedup_dist_sq = dedup_dist * dedup_dist;
                    bool has_nearby_confirmed = false;
                    for (const auto& pair : tracks_) {
                        if (pair.second.id == track.id) continue;
                        // 健全CONFIRMED航跡のみ: 瀕死航跡(misses>=3)は後継ブロックから除外
                        if (pair.second.track_state == TrackState::CONFIRMED &&
                            pair.second.misses < 3) {
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

        // TENTATIVE航跡: 確認ウィンドウ内にヒット蓄積できなければ削除
        // confirm_window基本、部分ヒットありなら1窓分延長
        if (track.track_state == TrackState::TENTATIVE) {
            int num_known = 0;
            for (const auto& pair : tracks_) {
                if (pair.second.track_state == TrackState::CONFIRMED &&
                    pair.second.misses < 3) num_known++;
            }
            int tent_confirm_hits = (num_known >= 4)
                ? params_.confirm_hits
                : std::min(params_.confirm_hits, 2);
            int effective_window = params_.confirm_window;
            if (track.hits > 0 && track.hits < tent_confirm_hits) {
                effective_window += params_.confirm_window;
            }
            if (track.age >= effective_window && track.hits < tent_confirm_hits) {
                track.track_state = TrackState::DELETED;
                return;
            }
        }

        // Change A: 低ヒットCONFIRMED航跡の早期削除
        // hits==2（最低確認ヒット数）かつmisses>=5の航跡は早期LOST化。
        // クラッタ由来の偽トラックを素早く排除し、リソースを解放する。
        if (track.track_state == TrackState::CONFIRMED &&
            track.hits == 2 && track.misses >= 5) {
            track.track_state = TrackState::LOST;
            return;
        }

        // CONFIRMED航跡の消失判定
        // 近接にCONFIRMED航跡がある場合、観測競合（GNNの1対1割り当て）により
        // 一方がBeamMissを蓄積して消滅→再生成のサイクルが発生する。
        // これを防止するため、近接航跡がある場合はdelete_missesを3倍に緩和する。
        if (track.track_state == TrackState::CONFIRMED) {
            int effective_delete_misses = params_.delete_misses;

            // ヒット数比例のdelete_misses緩和: 追尾実績に応じて検出ギャップ耐性を向上
            // hits/5 を加算（10ヒット→+2, 20ヒット→+4, 50ヒット→+10, 100ヒット→+20）
            // 最大2倍までキャップ
            int hits_bonus = track.hits / 5;
            effective_delete_misses = std::min(
                params_.delete_misses + hits_bonus,
                params_.delete_misses * 2);

            // 近接CONFIRMED航跡がある場合、観測競合による交互消失を防止
            if (params_.min_init_distance > 0.0f) {
                float proximity_sq = 10000.0f * 10000.0f;  // 10km
                for (const auto& pair : tracks_) {
                    if (pair.second.id == track.id) continue;
                    if (pair.second.track_state == TrackState::CONFIRMED) {
                        float dx = track.state(0) - pair.second.state(0);
                        float dy = track.state(1) - pair.second.state(1);
                        float dz = track.state(2) - pair.second.state(2);
                        if (dx*dx + dy*dy + dz*dz < proximity_sq) {
                            effective_delete_misses = std::max(effective_delete_misses,
                                                               params_.delete_misses * 2);
                            break;
                        }
                    }
                }
            }
            if (track.misses >= effective_delete_misses) {
                track.track_state = TrackState::LOST;
            }
        }

        // LOST航跡の最終削除
        // LOST航跡はデータアソシエーションに参加し観測を消費するため、
        // 長く残すと新規TENTATIVE生成を妨げ、OSPA/精度が悪化する。
        // 短い猶予で削除し測定リソースを早期解放する。
        if (track.track_state == TrackState::LOST) {
            int grace_misses = params_.delete_misses + 5;  // 15: delete_misses + 5フレーム猶予
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
        // CONFIRMED + TENTATIVE のみ（データアソシエーション参加対象）
        // LOST航跡は除外: 予測位置が不正確で観測を無駄に消費し、
        // 新規TENTATIVE生成を妨げる
        if (pair.second.track_state == TrackState::CONFIRMED ||
            pair.second.track_state == TrackState::TENTATIVE) {
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
