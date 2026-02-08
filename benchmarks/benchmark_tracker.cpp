#include <benchmark/benchmark.h>
#include "tracker/multi_target_tracker.hpp"
#include "simulation/target_generator.hpp"
#include "simulation/radar_simulator.hpp"

using namespace fasttracker;

// 多目標追尾のエンドツーエンドベンチマーク
static void BM_Tracker_EndToEnd(benchmark::State& state) {
    int num_targets = state.range(0);

    // シミュレーション環境のセットアップ
    TargetGenerator target_gen(num_targets);
    target_gen.initializeDefaultScenario();

    RadarSimulator radar_sim(target_gen);

    // トラッカー初期化
    MultiTargetTracker tracker(num_targets * 2);

    double time = 0.0;
    double dt = 1.0 / 30.0;  // 30 Hz

    // ウォームアップ
    auto meas = radar_sim.generate(time);
    tracker.update(meas, time);

    for (auto _ : state) {
        time += dt;
        auto measurements = radar_sim.generate(time);
        tracker.update(measurements, time);
    }

    state.SetItemsProcessed(state.iterations() * num_targets);

    // 最終性能を出力
    const auto& perf = tracker.getLastPerformanceStats();
    state.counters["FPS"] = benchmark::Counter(1000.0 / perf.total_time_ms);
    state.counters["Tracks"] = benchmark::Counter(tracker.getNumTracks());
    state.counters["Confirmed"] = benchmark::Counter(tracker.getNumConfirmedTracks());
}

// 予測ステップのみ
static void BM_Tracker_Predict(benchmark::State& state) {
    int num_targets = state.range(0);

    TargetGenerator target_gen(num_targets);
    target_gen.initializeDefaultScenario();

    RadarSimulator radar_sim(target_gen);
    MultiTargetTracker tracker(num_targets * 2);

    // 初期トラックを作成
    auto meas = radar_sim.generate(0.0);
    tracker.update(meas, 0.0);

    double time = 0.1;

    for (auto _ : state) {
        // 観測なしで予測のみ
        tracker.update({}, time);
        time += 0.1;
    }

    state.SetItemsProcessed(state.iterations() * tracker.getNumTracks());
}

// データアソシエーションのベンチマーク
static void BM_Tracker_Association(benchmark::State& state) {
    int num_targets = state.range(0);

    TargetGenerator target_gen(num_targets);
    target_gen.initializeDefaultScenario();

    RadarSimulator radar_sim(target_gen);
    MultiTargetTracker tracker(num_targets * 2);

    double time = 0.0;

    for (auto _ : state) {
        time += 0.1;
        auto measurements = radar_sim.generate(time);
        tracker.update(measurements, time);
    }

    const auto& perf = tracker.getLastPerformanceStats();
    state.counters["AssocTime_ms"] = benchmark::Counter(perf.association_time_ms);
}

// スケーラビリティテスト
BENCHMARK(BM_Tracker_EndToEnd)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Tracker_Predict)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Tracker_Association)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
