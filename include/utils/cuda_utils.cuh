#ifndef FASTTRACKER_CUDA_UTILS_CUH
#define FASTTRACKER_CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <memory>

namespace fasttracker {
namespace cuda {

// CUDAエラーチェックマクロ
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDAカーネル起動エラーチェック
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            throw std::runtime_error(std::string("CUDA kernel error: ") + \
                                     cudaGetErrorString(error)); \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

// デバイスメモリ管理用RAIIラッパー
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    explicit DeviceMemory(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        } else {
            ptr_ = nullptr;
        }
    }

    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // コピーコンストラクタ削除（所有権の移動のみ許可）
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // ムーブコンストラクタ
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // ホストからデバイスへコピー
    void copyFrom(const T* host_data, size_t count) {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds allocated size");
        }
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // デバイスからホストへコピー
    void copyTo(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds allocated size");
        }
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    // メモリをゼロ初期化
    void zero() {
        if (ptr_ && size_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_;
    size_t size_;
};

// カーネル起動設定の計算
struct KernelConfig {
    dim3 grid;
    dim3 block;

    KernelConfig(int total_threads, int block_size = 256) {
        block = dim3(block_size);
        grid = dim3((total_threads + block_size - 1) / block_size);
    }

    KernelConfig(int threads_x, int threads_y, int block_size = 16) {
        block = dim3(block_size, block_size);
        grid = dim3((threads_x + block_size - 1) / block_size,
                    (threads_y + block_size - 1) / block_size);
    }
};

// デバイス情報取得
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;

    static DeviceInfo getCurrent() {
        DeviceInfo info;
        CUDA_CHECK(cudaGetDevice(&info.device_id));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, info.device_id));

        info.name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.multiprocessor_count = prop.multiProcessorCount;
        info.max_threads_per_block = prop.maxThreadsPerBlock;
        info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        info.free_memory = free;

        return info;
    }

    void print() const {
        printf("=== CUDA Device Info ===\n");
        printf("Device ID: %d\n", device_id);
        printf("Name: %s\n", name.c_str());
        printf("Compute Capability: %d.%d\n",
               compute_capability_major, compute_capability_minor);
        printf("Total Memory: %.2f GB\n", total_memory / (1024.0 * 1024.0 * 1024.0));
        printf("Free Memory: %.2f GB\n", free_memory / (1024.0 * 1024.0 * 1024.0));
        printf("Multiprocessors: %d\n", multiprocessor_count);
        printf("Max Threads/Block: %d\n", max_threads_per_block);
        printf("Max Threads/MP: %d\n", max_threads_per_multiprocessor);
        printf("========================\n");
    }
};

// CUDAストリーム管理
class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t get() { return stream_; }

private:
    cudaStream_t stream_;
};

// タイマー（GPU時間測定）
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
    }

    float elapsed() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventSynchronize(stop_));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace cuda
} // namespace fasttracker

#endif // FASTTRACKER_CUDA_UTILS_CUH
