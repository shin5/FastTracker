#include <gtest/gtest.h>
#include "simulation/target_generator.hpp"
#include "simulation/radar_simulator.hpp"

using namespace fasttracker;

class SimulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        target_gen_ = std::make_unique<TargetGenerator>(10);
        target_gen_->initializeDefaultScenario();
    }

    std::unique_ptr<TargetGenerator> target_gen_;
};

TEST_F(SimulationTest, TargetGeneration) {
    EXPECT_EQ(target_gen_->getNumTargets(), 10);

    auto states = target_gen_->generateStates(0.0);
    EXPECT_EQ(states.size(), 10);

    // 全ての目標が状態ベクトルを持つことを確認
    for (const auto& state : states) {
        EXPECT_EQ(state.size(), STATE_DIM);
    }
}

TEST_F(SimulationTest, TargetMotion) {
    auto states_t0 = target_gen_->generateStates(0.0);
    auto states_t1 = target_gen_->generateStates(1.0);

    EXPECT_EQ(states_t0.size(), states_t1.size());

    // 1秒後の位置が変化していることを確認
    float total_displacement = 0.0f;
    for (size_t i = 0; i < states_t0.size(); i++) {
        float dx = states_t1[i](0) - states_t0[i](0);
        float dy = states_t1[i](1) - states_t0[i](1);
        total_displacement += std::sqrt(dx * dx + dy * dy);
    }

    EXPECT_GT(total_displacement, 0.0f);
}

TEST_F(SimulationTest, RadarMeasurements) {
    RadarSimulator radar_sim(*target_gen_);

    auto measurements = radar_sim.generate(0.0);

    // 観測が生成されることを確認
    EXPECT_GT(measurements.size(), 0);

    // 各観測が妥当な値を持つことを確認
    for (const auto& meas : measurements) {
        EXPECT_GT(meas.range, 0.0f);
        EXPECT_GE(meas.azimuth, -M_PI);
        EXPECT_LE(meas.azimuth, M_PI);
    }
}

TEST_F(SimulationTest, DetectionProbability) {
    RadarSimulator radar_sim(*target_gen_);

    // 検出確率を1.0に設定
    radar_sim.setDetectionProbability(1.0f);
    radar_sim.setClutterDensity(0.0f);  // クラッタなし

    auto measurements = radar_sim.generate(0.0);

    // 全ての目標が検出されるはず
    EXPECT_EQ(measurements.size(), target_gen_->getNumTargets());
}

TEST_F(SimulationTest, ClusterScenario) {
    TargetGenerator clustered_gen(50);
    clustered_gen.generateClusteredScenario(Eigen::Vector2f(0.0f, 0.0f), 500.0f);

    auto states = clustered_gen.generateStates(0.0);
    EXPECT_EQ(states.size(), 50);

    // 全ての目標がクラスタ内にあることを確認
    for (const auto& state : states) {
        float range = std::sqrt(state(0) * state(0) + state(1) * state(1));
        EXPECT_LT(range, 600.0f);  // 少し余裕を持たせる
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
