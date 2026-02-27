cmake_minimum_required(VERSION 3.18)
project(FastTracker LANGUAGES CXX CUDA)

# C++17標準を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ビルドタイプのデフォルト設定
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# コンパイラフラグ
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /utf-8")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

# CUDAアーキテクチャ設定（環境に応じて調整）
# RTX 3080/3090: 86, RTX 4080/4090: 89, V100: 70, A100: 80
set(CMAKE_CUDA_ARCHITECTURES "86;89")

# CUDAコンパイラフラグ
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# インクルードディレクトリ
include_directories(${PROJECT_SOURCE_DIR}/include)

# CUDA Toolkit検出
find_package(CUDAToolkit REQUIRED)

# Eigen3検出
set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/eigen-3.4.0")
if(EXISTS "${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Using local Eigen3 from: ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
else()
    find_package(Eigen3 3.3 REQUIRED)
endif()

# ソースファイル
set(UTILS_SOURCES
    src/utils/cuda_utils.cu
)

set(UKF_SOURCES
    src/ukf/ukf.cu
    src/ukf/ukf_kernels.cu
    src/ukf/imm_ukf.cpp
    src/ukf/imm_ukf.cu
)

set(TRACKER_SOURCES
    src/tracker/multi_target_tracker.cpp
    src/tracker/track_manager.cpp
    src/tracker/data_association.cu
    src/tracker/jpda_association.cpp
    src/tracker/pmbm_association.cpp
    src/tracker/mht_association.cpp
    src/tracker/glmb_association.cpp
)

set(SIMULATION_SOURCES
    src/simulation/radar_simulator.cpp
    src/simulation/target_generator.cpp
    src/simulation/target_generator_missiles.cpp
)

set(EVALUATION_SOURCES
    src/evaluation/tracking_evaluator.cpp
)

# FastTrackerライブラリ
add_library(fasttracker_lib STATIC
    ${UTILS_SOURCES}
    ${UKF_SOURCES}
    ${TRACKER_SOURCES}
    ${SIMULATION_SOURCES}
    ${EVALUATION_SOURCES}
)

target_link_libraries(fasttracker_lib
    CUDA::cudart
    CUDA::curand
    Eigen3::Eigen
)

# メインアプリケーション
add_executable(fasttracker
    src/main.cpp
)

target_link_libraries(fasttracker
    fasttracker_lib
)

# Google Test（テスト用）
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest)

    if(NOT GTest_FOUND)
        message(STATUS "GTest not found, attempting to download...")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(test_ukf tests/test_ukf.cpp)
    target_link_libraries(test_ukf fasttracker_lib GTest::gtest_main)
    add_test(NAME test_ukf COMMAND test_ukf)

    add_executable(test_tracker tests/test_tracker.cpp)
    target_link_libraries(test_tracker fasttracker_lib GTest::gtest_main)
    add_test(NAME test_tracker COMMAND test_tracker)

    add_executable(test_simulation tests/test_simulation.cpp)
    target_link_libraries(test_simulation fasttracker_lib GTest::gtest_main)
    add_test(NAME test_simulation COMMAND test_simulation)
endif()

# Google Benchmark（ベンチマーク用）
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    find_package(benchmark)

    if(NOT benchmark_FOUND)
        message(STATUS "Google Benchmark not found, attempting to download...")
        include(FetchContent)
        FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.8.3
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
    endif()

    add_executable(benchmark_ukf benchmarks/benchmark_ukf.cpp)
    target_link_libraries(benchmark_ukf fasttracker_lib benchmark::benchmark)

    add_executable(benchmark_tracker benchmarks/benchmark_tracker.cpp)
    target_link_libraries(benchmark_tracker fasttracker_lib benchmark::benchmark)
endif()

# インストール設定
install(TARGETS fasttracker DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# デバッグ情報の出力
message(STATUS "")
message(STATUS "FastTracker Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "")
