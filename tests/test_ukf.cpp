#include <gtest/gtest.h>
#include "ukf/ukf.cuh"
#include "utils/types.hpp"

using namespace fasttracker;

class UKFTest : public ::testing::Test {
protected:
    void SetUp() override {
        ukf_ = std::make_unique<UKF>(10);
    }

    std::unique_ptr<UKF> ukf_;
};

TEST_F(UKFTest, Initialization) {
    EXPECT_EQ(ukf_->getMaxTargets(), 10);
    EXPECT_NEAR(ukf_->getParams().alpha, 0.001f, 1e-6f);
    EXPECT_NEAR(ukf_->getParams().beta, 2.0f, 1e-6f);
}

TEST_F(UKFTest, PredictSingleTarget) {
    // 単一目標の予測テスト
    StateVector state;
    state << 100.0f, 200.0f, 10.0f, 20.0f, 0.0f, 0.0f;

    StateCov cov = StateCov::Identity() * 100.0f;

    std::vector<StateVector> states = {state};
    std::vector<StateCov> covs = {cov};

    ukf_->copyToDevice(states, covs);
    ukf_->predict(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(), 1, 1.0f);
    ukf_->copyToHost(states, covs, 1);

    // 1秒後の予測: x = 100 + 10*1 = 110, y = 200 + 20*1 = 220
    EXPECT_NEAR(states[0](0), 110.0f, 1.0f);
    EXPECT_NEAR(states[0](1), 220.0f, 1.0f);
}

TEST_F(UKFTest, UpdateWithMeasurement) {
    // 観測による更新テスト
    StateVector state;
    state << 100.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f;

    StateCov cov = StateCov::Identity() * 100.0f;

    // 観測: range=100, azimuth=0, doppler=10
    std::vector<float> measurements = {100.0f, 0.0f, 0.0f, 10.0f};

    std::vector<StateVector> states = {state};
    std::vector<StateCov> covs = {cov};

    ukf_->copyToDevice(states, covs);

    // 観測のコピー（デバイスメモリ）
    cuda::DeviceMemory<float> d_meas(4);
    d_meas.copyFrom(measurements.data(), 4);

    ukf_->update(ukf_->getDeviceStates(), ukf_->getDeviceCovariances(),
                 d_meas.get(), 1);

    ukf_->copyToHost(states, covs, 1);

    // 更新後、共分散が減少することを確認
    EXPECT_LT(covs[0].trace(), cov.trace());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
