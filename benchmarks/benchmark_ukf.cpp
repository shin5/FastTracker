#include <benchmark/benchmark.h>
#include "ukf/ukf.cuh"
#include "utils/types.hpp"
#include <random>

using namespace fasttracker;

// UKF予測ステップのベンチマーク
static void BM_UKF_Predict(benchmark::State& state) {
    int num_targets = state.range(0);

    UKF ukf(num_targets);

    // ランダムな初期状態を生成
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<StateVector> states(num_targets);
    std::vector<StateCov> covs(num_targets);

    for (int i = 0; i < num_targets; i++) {
        states[i] << dist(rng), dist(rng), dist(rng) * 0.1f,
                     dist(rng) * 0.1f, 0.0f, 0.0f;
        covs[i] = StateCov::Identity() * 100.0f;
    }

    ukf.copyToDevice(states, covs);

    // ウォームアップ
    ukf.predict(ukf.getDeviceStates(), ukf.getDeviceCovariances(), num_targets, 0.1f);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        ukf.predict(ukf.getDeviceStates(), ukf.getDeviceCovariances(), num_targets, 0.1f);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * num_targets);
}

// UKF更新ステップのベンチマーク
static void BM_UKF_Update(benchmark::State& state) {
    int num_targets = state.range(0);

    UKF ukf(num_targets);

    // ランダムな初期状態と観測を生成
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<StateVector> states(num_targets);
    std::vector<StateCov> covs(num_targets);
    std::vector<float> measurements(num_targets * MEAS_DIM);

    for (int i = 0; i < num_targets; i++) {
        states[i] << dist(rng), dist(rng), dist(rng) * 0.1f,
                     dist(rng) * 0.1f, 0.0f, 0.0f;
        covs[i] = StateCov::Identity() * 100.0f;

        float x = states[i](0);
        float y = states[i](1);
        measurements[i * MEAS_DIM + 0] = std::sqrt(x * x + y * y);
        measurements[i * MEAS_DIM + 1] = std::atan2(y, x);
        measurements[i * MEAS_DIM + 2] = 0.0f;
        measurements[i * MEAS_DIM + 3] = 0.0f;
    }

    ukf.copyToDevice(states, covs);

    cuda::DeviceMemory<float> d_meas(num_targets * MEAS_DIM);
    d_meas.copyFrom(measurements.data(), num_targets * MEAS_DIM);

    // ウォームアップ
    ukf.update(ukf.getDeviceStates(), ukf.getDeviceCovariances(),
               d_meas.get(), num_targets);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        ukf.update(ukf.getDeviceStates(), ukf.getDeviceCovariances(),
                   d_meas.get(), num_targets);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * num_targets);
}

// スケーラビリティテスト
BENCHMARK(BM_UKF_Predict)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_UKF_Update)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
