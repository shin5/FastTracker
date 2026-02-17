#include <gtest/gtest.h>
#include "tracker/multi_target_tracker.hpp"
#include "utils/types.hpp"

using namespace fasttracker;

class TrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tracker_ = std::make_unique<MultiTargetTracker>(100);
    }

    std::unique_ptr<MultiTargetTracker> tracker_;
};

TEST_F(TrackerTest, Initialization) {
    EXPECT_EQ(tracker_->getNumTracks(), 0);
    EXPECT_EQ(tracker_->getNumConfirmedTracks(), 0);
}

TEST_F(TrackerTest, InitializeTracksFromMeasurements) {
    // 初期観測からトラック生成
    std::vector<Measurement> measurements;

    for (int i = 0; i < 5; i++) {
        Measurement meas;
        meas.range = 1000.0f + i * 100.0f;
        meas.azimuth = 0.0f;
        meas.elevation = 0.0f;
        meas.doppler = 10.0f;
        meas.timestamp = 0.0;
        measurements.push_back(meas);
    }

    tracker_->update(measurements, 0.0);

    EXPECT_EQ(tracker_->getNumTracks(), 5);
}

TEST_F(TrackerTest, TrackConfirmation) {
    // トラック確定のテスト
    Measurement meas;
    meas.range = 1000.0f;
    meas.azimuth = 0.0f;
    meas.elevation = 0.0f;
    meas.doppler = 10.0f;
    meas.timestamp = 0.0;

    // 初回
    tracker_->update({meas}, 0.0);
    EXPECT_EQ(tracker_->getNumConfirmedTracks(), 0);  // まだ仮トラック

    // 複数回観測
    for (int i = 1; i <= 5; i++) {
        meas.timestamp = i * 0.1;
        tracker_->update({meas}, i * 0.1);
    }

    // 3/5ルールで確定されるはず
    EXPECT_GT(tracker_->getNumConfirmedTracks(), 0);
}

TEST_F(TrackerTest, TrackDeletion) {
    // トラック削除のテスト
    Measurement meas;
    meas.range = 1000.0f;
    meas.azimuth = 0.0f;
    meas.elevation = 0.0f;
    meas.doppler = 10.0f;
    meas.timestamp = 0.0;

    tracker_->update({meas}, 0.0);
    EXPECT_EQ(tracker_->getNumTracks(), 1);

    // 観測なしで複数回更新
    for (int i = 1; i <= 10; i++) {
        tracker_->update({}, i * 0.1);
    }

    // 5フレーム未検出で削除されるはず
    EXPECT_EQ(tracker_->getNumTracks(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
