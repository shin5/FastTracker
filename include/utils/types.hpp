#ifndef FASTTRACKER_TYPES_HPP
#define FASTTRACKER_TYPES_HPP

#include <vector>
#include <memory>
#include <Eigen/Dense>

// M_PI定義（Windowsで必要）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

// 状態ベクトル次元 [x, y, vx, vy, ax, ay]
constexpr int STATE_DIM = 6;

// 観測ベクトル次元 [range, azimuth, elevation, doppler]
constexpr int MEAS_DIM = 4;

// UKFシグマポイント数 (2n+1)
constexpr int SIGMA_POINTS = 2 * STATE_DIM + 1;

// 状態ベクトル型
using StateVector = Eigen::Matrix<float, STATE_DIM, 1>;

// 観測ベクトル型
using MeasVector = Eigen::Matrix<float, MEAS_DIM, 1>;

// 状態共分散行列型
using StateCov = Eigen::Matrix<float, STATE_DIM, STATE_DIM>;

// 観測共分散行列型
using MeasCov = Eigen::Matrix<float, MEAS_DIM, MEAS_DIM>;

// シグマポイント行列型
using SigmaPoints = Eigen::Matrix<float, STATE_DIM, SIGMA_POINTS>;

// レーダー観測データ
struct Measurement {
    float range;        // 距離 [m]
    float azimuth;      // 方位角 [rad]
    float elevation;    // 仰角 [rad]
    float doppler;      // ドップラー速度 [m/s]
    float snr;          // SN比 [dB]
    double timestamp;   // タイムスタンプ [s]
    int sensor_id;      // センサーID
    bool is_clutter;    // クラッタ（誤警報）フラグ

    Measurement() : range(0.0f), azimuth(0.0f), elevation(0.0f),
                    doppler(0.0f), snr(0.0f), timestamp(0.0), sensor_id(0),
                    is_clutter(false) {}
};

// トラック状態
enum class TrackState {
    TENTATIVE,      // 仮トラック
    CONFIRMED,      // 確定トラック
    LOST            // 消失トラック
};

// トラック情報
struct Track {
    int id;                     // トラックID
    StateVector state;          // 状態推定値
    StateCov covariance;        // 共分散行列
    TrackState track_state;     // トラック状態
    int hits;                   // 観測ヒット数
    int misses;                 // 観測ミス数
    double last_update_time;    // 最終更新時刻
    std::vector<float> model_probs;  // IMMモデル確率（3モデル用）

    Track() : id(-1), hits(0), misses(0), last_update_time(0.0),
              track_state(TrackState::TENTATIVE) {
        state.setZero();
        covariance.setIdentity();
        model_probs = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};  // 初期均等確率
    }
};

// UKFパラメータ
struct UKFParams {
    float alpha;    // スケーリングパラメータ (0.001)
    float beta;     // 分布形状パラメータ (2.0 for Gaussian)
    float kappa;    // 補助パラメータ (0.0)
    float lambda;   // 複合スケーリングパラメータ

    UKFParams() : alpha(0.5f), beta(2.0f), kappa(0.0f) {
        lambda = alpha * alpha * (STATE_DIM + kappa) - STATE_DIM;
    }
};

// データアソシエーションパラメータ
struct AssociationParams {
    float gate_threshold;       // ゲート閾値（Mahalanobis距離）
    float max_distance;         // 最大許容距離
    int confirm_hits;           // トラック確定に必要なヒット数
    int confirm_window;         // トラック確定判定ウィンドウ
    int delete_misses;          // トラック削除閾値
    float min_snr_for_init;     // トラック生成に必要な最小SNR [dB]

    AssociationParams()
        : gate_threshold(500.0f),
          max_distance(100.0f),
          confirm_hits(2),
          confirm_window(5),
          delete_misses(20),
          min_snr_for_init(22.0f) {}
};

// プロセスノイズ
struct ProcessNoise {
    float position_noise;   // 位置ノイズ std [m]
    float velocity_noise;   // 速度ノイズ std [m/s]
    float accel_noise;      // 加速度ノイズ std [m/s²]

    ProcessNoise()
        : position_noise(160.0f),
          velocity_noise(65.0f),
          accel_noise(160.0f) {}
};

// 観測ノイズ
struct MeasurementNoise {
    float range_noise;      // 距離ノイズ std [m]
    float azimuth_noise;    // 方位角ノイズ std [rad]
    float elevation_noise;  // 仰角ノイズ std [rad]
    float doppler_noise;    // ドップラーノイズ std [m/s]

    MeasurementNoise()
        : range_noise(10.0f),
          azimuth_noise(0.01f),   // ~0.57度
          elevation_noise(0.01f),  // ~0.57度
          doppler_noise(2.0f) {}
};

// GPU用のフラット配列データ構造（メモリ効率向上）
struct GPUTrackData {
    float* states;          // [num_tracks * STATE_DIM]
    float* covariances;     // [num_tracks * STATE_DIM * STATE_DIM]
    int* track_ids;         // [num_tracks]
    int num_tracks;
};

struct GPUMeasurementData {
    float* measurements;    // [num_meas * MEAS_DIM]
    double* timestamps;     // [num_meas]
    int* sensor_ids;        // [num_meas]
    int num_measurements;
};

} // namespace fasttracker

#endif // FASTTRACKER_TYPES_HPP
