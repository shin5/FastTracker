#include <iostream>
#include <fstream>
#include <iomanip>
#include "tracker/multi_target_tracker.hpp"
#include "simulation/target_generator.hpp"
#include "simulation/radar_simulator.hpp"
#include "utils/cuda_utils.cuh"

using namespace fasttracker;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --num-targets <N>    Number of targets (default: 1000)" << std::endl;
    std::cout << "  --duration <T>       Simulation duration in seconds (default: 10.0)" << std::endl;
    std::cout << "  --framerate <FPS>    Frame rate in Hz (default: 30)" << std::endl;
    std::cout << "  --output <file>      Output file for results (default: results.csv)" << std::endl;
    std::cout << "  --scenario <type>    Scenario type: default|clustered|high-maneuver (default: default)" << std::endl;
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
    } else {
        std::cerr << "Unknown scenario: " << scenario << std::endl;
        return 1;
    }

    RadarSimulator radar_sim(target_gen);

    // トラッカーの初期化
    MultiTargetTracker tracker(num_targets * 2);  // 余裕を持たせる

    // シミュレーションループ
    double dt = 1.0 / framerate;
    int num_frames = static_cast<int>(duration / dt);

    std::cout << "\nStarting simulation..." << std::endl;
    std::cout << "Total frames: " << num_frames << std::endl;

    // 結果ファイルのオープン
    std::ofstream out_file(output_file);
    out_file << "frame,time,num_tracks,num_confirmed,num_measurements,processing_time_ms" << std::endl;

    for (int frame = 0; frame < num_frames; frame++) {
        double current_time = frame * dt;

        // 観測データ生成
        auto measurements = radar_sim.generate(current_time);

        // トラッカー更新
        tracker.update(measurements, current_time);

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

    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "Results saved to: " << output_file << std::endl;

    // 最終統計
    std::cout << "\n=== Final Statistics ===" << std::endl;
    tracker.printStatistics();
    radar_sim.printStatistics();

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
