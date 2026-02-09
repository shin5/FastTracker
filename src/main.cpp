#pragma execution_character_set("utf-8")

#include <iostream>
#include <fstream>
#include <iomanip>
#include "tracker/multi_target_tracker.hpp"
#include "simulation/target_generator.hpp"
#include "simulation/radar_simulator.hpp"
#include "evaluation/tracking_evaluator.hpp"
#include "utils/cuda_utils.cuh"

using namespace fasttracker;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --num-targets <N>    Number of targets (default: 1000)" << std::endl;
    std::cout << "  --duration <T>       Simulation duration in seconds (default: 10.0)" << std::endl;
    std::cout << "  --framerate <FPS>    Frame rate in Hz (default: 30)" << std::endl;
    std::cout << "  --output <file>      Output file for results (default: results.csv)" << std::endl;
    std::cout << "  --scenario <type>    Scenario type:" << std::endl;
    std::cout << "                         default         - Standard aircraft" << std::endl;
    std::cout << "                         clustered       - Dense targets" << std::endl;
    std::cout << "                         high-maneuver   - High maneuverability" << std::endl;
    std::cout << "                         ballistic       - Ballistic missiles" << std::endl;
    std::cout << "                         hypersonic      - Hypersonic glide vehicles" << std::endl;
    std::cout << "                         mixed-threat    - Mixed threats" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    // デフォルトパラメータ
    int num_targets = 1000;
    double duration = 10.0;
    double framerate = 30.0;
    std::string output_file = "results.csv";
    std::string scenario = "default";

    // コマンドライン引数解析
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--num-targets" && i + 1 < argc) {
            num_targets = std::atoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::atof(argv[++i]);
        } else if (arg == "--framerate" && i + 1 < argc) {
            framerate = std::atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--scenario" && i + 1 < argc) {
            scenario = argv[++i];
        }
    }

    std::cout << "=== FastTracker: GPU-Accelerated Multi-Target Tracker ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Targets: " << num_targets << std::endl;
    std::cout << "  Duration: " << duration << " s" << std::endl;
    std::cout << "  Frame rate: " << framerate << " Hz" << std::endl;
    std::cout << "  Scenario: " << scenario << std::endl;
    std::cout << "==========================================================" << std::endl;

    // CUDA デバイス情報
    try {
        auto device_info = cuda::DeviceInfo::getCurrent();
        device_info.print();
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
        return 1;
    }

    // シミュレーション環境のセットアップ
    TargetGenerator target_gen(num_targets);

    if (scenario == "default") {
        target_gen.initializeDefaultScenario();
    } else if (scenario == "clustered") {
        target_gen.generateClusteredScenario(Eigen::Vector2f(0.0f, 0.0f), 500.0f);
    } else if (scenario == "high-maneuver") {
        target_gen.generateHighManeuverScenario();
    } else if (scenario == "ballistic") {
        std::cout << "Ballistic Missile Scenario" << std::endl;
        target_gen.generateBallisticMissileScenario(num_targets);
    } else if (scenario == "hypersonic") {
        std::cout << "Hypersonic Glide Vehicle Scenario" << std::endl;
        target_gen.generateHypersonicGlideScenario(num_targets);
    } else if (scenario == "mixed-threat") {
        std::cout << "Mixed Threat Scenario (Aircraft + Missiles)" << std::endl;
        target_gen.generateMixedThreatScenario();
    } else {
        std::cerr << "Unknown scenario: " << scenario << std::endl;
        return 1;
    }

    // レーダーパラメータの設定（ミサイルシナリオでは高出力レーダーをシミュレート）
    RadarParameters radar_params;
    if (scenario == "ballistic" || scenario == "hypersonic" || scenario == "mixed-threat") {
        // 長距離監視用パラメータ
        radar_params.max_range = 150000.0f;      // 150km監視範囲
        radar_params.false_alarm_rate = 1e-10f;  // 誤警報率を下げる
        radar_params.snr_ref = 110.0f;           // 高出力レーダー（60dB → 110dB）
    }

    RadarSimulator radar_sim(target_gen, radar_params);

    // トラッカーパラメータの設定（ミサイルシナリオでは適応型設定）
    ProcessNoise process_noise;
    AssociationParams assoc_params;

    if (scenario == "ballistic" || scenario == "hypersonic" || scenario == "mixed-threat") {
        // 弾道ミサイル用: 最適化済みプロセスノイズ（第2弾チューニング）
        process_noise.position_noise = 4.0f;     // 5 → 4（さらなるノイズ削減）
        process_noise.velocity_noise = 8.0f;     // 10 → 8（予測範囲を適正化）
        process_noise.accel_noise = 4.0f;        // 5 → 4（IMMによる適応的対応）

        // 弾道ミサイル用: バランス型データアソシエーション（第2弾）
        assoc_params.gate_threshold = 15.0f;     // 18 → 15（やや緩和、確定率向上）
        assoc_params.max_distance = 4.0f;        // 5 → 4（距離制限強化）
        assoc_params.confirm_hits = 3;           // 4 → 3（現実的な確定条件）
        assoc_params.confirm_window = 4;         // 5 → 4（4フレーム中3回観測）
        assoc_params.delete_misses = 8;          // 6 → 8（確定の猶予期間確保）
        assoc_params.min_snr_for_init = 40.0f;   // 35 → 40dB（極めて厳格、真の目標のみ）
    }

    // トラッカーの初期化
    MultiTargetTracker tracker(num_targets * 2, UKFParams(), assoc_params, process_noise);

    // 評価器の初期化
    TrackingEvaluator evaluator(100.0f, 2);  // OSPA: cutoff=100m, order=2

    // シミュレーションループ
    double dt = 1.0 / framerate;
    int num_frames = static_cast<int>(duration / dt);

    std::cout << "\nStarting simulation..." << std::endl;
    std::cout << "Total frames: " << num_frames << std::endl;

    // 結果ファイルのオープン
    std::ofstream out_file(output_file);
    out_file << "frame,time,num_tracks,num_confirmed,num_measurements,processing_time_ms" << std::endl;

    // トラック詳細ファイルのオープン
    std::ofstream track_file("track_details.csv");
    track_file << "frame,time,track_id,x,y,vx,vy,ax,ay,state,model_prob_cv,model_prob_high,model_prob_med" << std::endl;

    // 真値軌道ファイルのオープン
    std::ofstream ground_truth_file("ground_truth.csv");
    ground_truth_file << "frame,time,target_id,x,y,vx,vy,ax,ay" << std::endl;

    for (int frame = 0; frame < num_frames; frame++) {
        double current_time = frame * dt;

        // 真値状態を取得（評価用）
        auto ground_truth = radar_sim.getTrueStates(current_time);

        // 真値を出力
        for (size_t i = 0; i < ground_truth.size(); i++) {
            const auto& gt = ground_truth[i];
            ground_truth_file << frame << ","
                            << current_time << ","
                            << i << ","
                            << gt(0) << ","  // x
                            << gt(1) << ","  // y
                            << gt(2) << ","  // vx
                            << gt(3) << ","  // vy
                            << gt(4) << ","  // ax
                            << gt(5) << std::endl;  // ay
        }

        // 観測データ生成
        auto measurements = radar_sim.generate(current_time);

        // トラッカー更新
        try {
            tracker.update(measurements, current_time);
        } catch (const std::exception& e) {
            std::cerr << "Exception in tracker.update() at frame " << frame
                      << ": " << e.what() << std::endl;
            return 1;
        }

        // CUDAエラーチェック
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cerr << "CUDA Error after frame " << frame << ": "
                      << cudaGetErrorString(cuda_err) << std::endl;
            return 1;
        }

        // 評価器更新
        auto tracks = tracker.getAllTracks();
        evaluator.update(tracks, ground_truth,
                        static_cast<int>(measurements.size()),
                        current_time);

        // トラック詳細を出力
        for (const auto& track : tracks) {
            // トラック状態を文字列化
            int state_value = 0;  // TENTATIVE
            if (track.track_state == TrackState::CONFIRMED) {
                state_value = 1;
            } else if (track.track_state == TrackState::LOST) {
                state_value = 2;
            }

            // IMMモデル確率（3モデル）
            float prob_cv = track.model_probs.size() >= 1 ? track.model_probs[0] : 0.333f;
            float prob_high = track.model_probs.size() >= 2 ? track.model_probs[1] : 0.333f;
            float prob_med = track.model_probs.size() >= 3 ? track.model_probs[2] : 0.333f;

            track_file << frame << ","
                      << current_time << ","
                      << track.id << ","
                      << track.state(0) << ","  // x
                      << track.state(1) << ","  // y
                      << track.state(2) << ","  // vx
                      << track.state(3) << ","  // vy
                      << track.state(4) << ","  // ax
                      << track.state(5) << ","  // ay
                      << state_value << ","
                      << prob_cv << ","
                      << prob_high << ","
                      << prob_med << std::endl;
        }

        // 統計情報
        const auto& perf = tracker.getLastPerformanceStats();

        // 結果を出力
        out_file << frame << ","
                 << current_time << ","
                 << tracker.getNumTracks() << ","
                 << tracker.getNumConfirmedTracks() << ","
                 << measurements.size() << ","
                 << perf.total_time_ms << std::endl;

        // プログレス表示
        if ((frame + 1) % 10 == 0 || frame == 0) {
            std::cout << "Frame " << (frame + 1) << "/" << num_frames
                      << " | Tracks: " << tracker.getNumConfirmedTracks()
                      << " | Meas: " << measurements.size()
                      << " | Time: " << std::fixed << std::setprecision(2)
                      << perf.total_time_ms << " ms"
                      << " (" << (1000.0 / perf.total_time_ms) << " FPS)" << std::endl;
        }
    }

    out_file.close();
    track_file.close();
    ground_truth_file.close();

    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "Results saved to: " << output_file << std::endl;
    std::cout << "Track details saved to: track_details.csv" << std::endl;
    std::cout << "Ground truth saved to: ground_truth.csv" << std::endl;

    // 最終統計
    std::cout << "\n=== Final Statistics ===" << std::endl;
    tracker.printStatistics();
    radar_sim.printStatistics();

    // 評価結果
    evaluator.printSummary();
    evaluator.exportToCSV("evaluation_results.csv");

    // 性能サマリー
    const auto& final_perf = tracker.getLastPerformanceStats();
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Average processing time: "
              << std::fixed << std::setprecision(2)
              << final_perf.total_time_ms << " ms/frame" << std::endl;
    std::cout << "Average throughput: "
              << (1000.0 / final_perf.total_time_ms) << " FPS" << std::endl;
    std::cout << "Targets per second: "
              << (tracker.getNumTracks() * 1000.0 / final_perf.total_time_ms) << std::endl;

    // 性能目標との比較
    std::cout << "\n=== Performance Goals ===" << std::endl;
    double target_fps = 30.0;
    double achieved_fps = 1000.0 / final_perf.total_time_ms;

    std::cout << "Target: " << num_targets << " targets @ " << target_fps << " Hz" << std::endl;
    std::cout << "Achieved: " << tracker.getNumTracks() << " targets @ "
              << std::fixed << std::setprecision(1) << achieved_fps << " Hz" << std::endl;

    if (achieved_fps >= target_fps) {
        std::cout << "✓ Performance goal ACHIEVED!" << std::endl;
    } else {
        std::cout << "✗ Performance goal not met (need to optimize)" << std::endl;
    }

    return 0;
}
