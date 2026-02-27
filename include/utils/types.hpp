#ifndef FASTTRACKER_TYPES_HPP
#define FASTTRACKER_TYPES_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <Eigen/Dense>

// M_PI定義（Windowsで必要）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fasttracker {

// 状態ベクトル次元 [x, y, z, vx, vy, vz, ax, ay, az]
constexpr int STATE_DIM = 9;

// 運動モデルID
enum class MotionModelID : int {
    CA = 0,                // 等加速度（汎用持続加速：ブースト/グライド）
    BALLISTIC = 1,         // 弾道（重力+大気抗力）
    COORDINATED_TURN = 2,  // 旋回（HGV機動）
    SKIP_GLIDE = 3         // スキップ/グライド（弾道+空力揚力）
};

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

// レーダー観測データ（センサーが生成する生の観測値）
// 注意: この構造体には Ground Truth 情報を含めない
// センサーは観測値が真の目標由来かクラッタ由来かを知らない
struct Measurement {
    float range;        // 距離 [m]
    float azimuth;      // 方位角 [rad]
    float elevation;    // 仰角 [rad]
    float doppler;      // ドップラー速度 [m/s]
    float snr;          // SN比 [dB]
    double timestamp;   // タイムスタンプ [s]
    int sensor_id;      // センサーID

    Measurement() : range(0.0f), azimuth(0.0f), elevation(0.0f),
                    doppler(0.0f), snr(0.0f), timestamp(0.0), sensor_id(0) {}
};

// Ground Truth 関連付け（評価専用）
// センサーやトラッカーからはアクセスできない設計とする
struct GroundTruthAssociation {
    int measurement_index;  // 観測値のインデックス（measurements配列内）
    int true_target_id;     // 真の目標ID（-1 = クラッタ）

    GroundTruthAssociation() : measurement_index(-1), true_target_id(-1) {}
    GroundTruthAssociation(int meas_idx, int target_id)
        : measurement_index(meas_idx), true_target_id(target_id) {}
};

// トラック状態
enum class TrackState {
    TENTATIVE,      // 仮トラック
    CONFIRMED,      // 確定トラック
    LOST,           // 消失トラック（再捕捉猶予中：データアソシエーションに参加）
    DELETED         // 完全削除（pruneで除去）
};

// トラック情報
struct Track {
    int id;                     // トラックID
    StateVector state;          // 状態推定値
    StateCov covariance;        // 共分散行列
    TrackState track_state;     // トラック状態
    int hits;                   // 観測ヒット数
    int misses;                 // 連続観測ミス数
    int age;                    // 生成からの総フレーム数（M-of-N確認用）
    double last_update_time;    // 最終更新時刻
    float snr_sum;              // 累積SNR [dB]（確認品質チェック用）
    float existence_prob;       // ベルヌーイ存在確率（PMBM用、GNN/JPDAでは未使用）
    std::vector<float> model_probs;  // IMMモデル確率（3モデル用）

    Track() : id(-1), hits(0), misses(0), age(0), last_update_time(0.0),
              snr_sum(0.0f), existence_prob(0.5f), track_state(TrackState::TENTATIVE) {
        state.setZero();
        covariance.setIdentity();
        // 均等な初期確率（モデルバイアスなし、データから学習）
        model_probs = {0.25f, 0.25f, 0.25f, 0.25f};  // CA, Ballistic, CT, SkipGlide
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

// データアソシエーション手法
enum class AssociationMethod {
    GNN,    // Global Nearest Neighbor (Hungarian)
    JPDA,   // Joint Probabilistic Data Association
    MHT,    // Multiple Hypothesis Tracking
    PMBM,   // Poisson Multi-Bernoulli Mixture
    GLMB    // Generalized Labeled Multi-Bernoulli
};

// GLMBサンプラー手法
enum class GLMBSampler {
    MURTY,  // Murty's K-best (deterministic, O(n³K))
    GIBBS   // Gibbs sampler (stochastic, O(n·m·sweeps))
};

// データアソシエーションパラメータ
struct AssociationParams {
    float gate_threshold;       // ゲート閾値（Mahalanobis距離）
    float max_distance;         // 最大許容距離
    int confirm_hits;           // トラック確定に必要なヒット数
    int confirm_window;         // トラック確定判定ウィンドウ
    int delete_misses;          // トラック削除閾値
    float min_snr_for_init;     // トラック生成に必要な最小SNR [dB]
    float max_jump_velocity;    // 位置ジャンプ制限用の最大速度 [m/s] (0=無制限)
    float min_init_distance;    // 新規トラック生成時の既存航跡との最小距離 [m] (0=チェックなし)

    AssociationMethod association_method = AssociationMethod::GNN;
    float jpda_pd = 0.8f;              // JPDA検出確率 Pd
    float jpda_clutter_density = 1e-6f; // JPDAクラッタ密度 λ
    float jpda_gate = 15.0f;           // JPDAゲート閾値（正規化距離）

    // PMBM パラメータ
    float pmbm_pd = 0.85f;              // PMBM検出確率
    float pmbm_gate = 50.0f;            // PMBMゲート閾値（正規化距離）
    float pmbm_clutter_density = 1e-6f; // PMBMクラッタ密度
    int   pmbm_k_best = 5;              // Murty's K-best仮説数
    float pmbm_survival_prob = 0.99f;   // 生存確率
    float pmbm_initial_existence = 0.2f;// 新規トラック初期存在確率
    float pmbm_confirm_existence = 0.5f;// CONFIRMED遷移閾値
    float pmbm_prune_existence = 0.01f; // 削除閾値

    // MHT パラメータ
    float mht_pd = 0.85f;              // MHT検出確率
    float mht_gate = 50.0f;            // MHTゲート閾値（正規化距離）
    float mht_clutter_density = 1e-6f; // MHTクラッタ密度
    int   mht_k_best = 10;             // Murty's K-best仮説数（フレームあたり）
    int   mht_max_hypotheses = 100;    // 最大グローバル仮説数 M
    float mht_score_decay = 0.9f;      // フレーム間スコア減衰率
    float mht_prune_ratio = 0.01f;     // プルーニング閾値比（ベスト比）
    float mht_switch_cost = 0.0f;      // 割当切替コスト（0=無効, >0でposition-based consistency有効）

    // GLMB パラメータ
    float glmb_pd = 0.85f;              // GLMB検出確率
    float glmb_gate = 50.0f;            // GLMBゲート閾値（正規化距離）
    float glmb_clutter_density = 1e-6f; // GLMBクラッタ密度
    int   glmb_k_best = 5;             // Murty's K-best仮説数（フレームあたり）
    int   glmb_max_hypotheses = 50;    // 最大グローバル仮説数 M
    float glmb_survival_prob = 0.99f;  // 生存確率 Ps
    float glmb_birth_weight = 0.01f;   // 誕生強度（新ラベル重み）
    float glmb_prune_weight = 1e-5f;   // 仮説プルーニング閾値
    float glmb_initial_existence = 0.2f; // 新規トラック初期存在確率
    float glmb_confirm_existence = 0.5f; // CONFIRMED遷移閾値
    float glmb_prune_existence = 0.01f;  // 削除閾値
    float glmb_score_decay = 0.9f;      // フレーム間スコア減衰率
    GLMBSampler glmb_sampler = GLMBSampler::MURTY;  // サンプラー手法
    int   glmb_gibbs_sweeps = 100;     // Gibbs: 総スイープ数
    int   glmb_gibbs_burnin = 10;      // Gibbs: バーンイン期間

    AssociationParams()
        : gate_threshold(500.0f),
          max_distance(100.0f),
          confirm_hits(2),
          confirm_window(5),
          delete_misses(90),       // 30Hzで3秒 — 弾道追尾での一時的検出欠落に対応
          min_snr_for_init(22.0f),
          max_jump_velocity(10000.0f),  // 10 km/s (HGV/弾道ミサイル上限)
          min_init_distance(0.0f) {}    // 0=無制限（デフォルト）
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

// 状態ベクトル・共分散の妥当性チェック
// NaN/Inf または物理的に有り得ない値を検出したら false を返す
// 異常値と判定された場合は直前フレームの値を維持することで伝播を防ぐ
inline bool isStateValid(const StateVector& s, const StateCov& cov) {
    // NaN / Inf チェック（全9要素）— 最初に最安コストで除外
    for (int i = 0; i < STATE_DIM; i++) {
        if (!std::isfinite(s(i))) return false;
    }
    // 物理的上限（弾道ミサイル追尾の現実的範囲に対する余裕倍）
    constexpr float MAX_POS_M   = 1.0e7f;   // 位置: ±10,000 km（地球直径の約80%）
    constexpr float MAX_VEL_MS  = 1.0e4f;   // 速度: ±10 km/s（低軌道速度 7.9 km/s 超え）
    constexpr float MAX_ACC_MS2 = 2000.0f;  // 加速度: ±2000 m/s²（約200G）
    if (std::abs(s(0)) > MAX_POS_M  || std::abs(s(1)) > MAX_POS_M  || std::abs(s(2)) > MAX_POS_M)  return false;
    if (std::abs(s(3)) > MAX_VEL_MS || std::abs(s(4)) > MAX_VEL_MS || std::abs(s(5)) > MAX_VEL_MS) return false;
    if (std::abs(s(6)) > MAX_ACC_MS2 || std::abs(s(7)) > MAX_ACC_MS2 || std::abs(s(8)) > MAX_ACC_MS2) return false;
    // 共分散対角: 有限かつ非負（負の分散は数値崩壊の兆候）
    for (int i = 0; i < STATE_DIM; i++) {
        if (!std::isfinite(cov(i, i)) || cov(i, i) < 0.0f) return false;
    }
    return true;
}

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
