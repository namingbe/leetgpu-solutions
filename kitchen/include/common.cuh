#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CHECK_LAST_CUDA_ERROR() CHECK_CUDA(cudaGetLastError())

class CudaTimer {
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

// Grid-stride loop helper
template<typename T>
__device__ inline void grid_stride_loop(T start, T end, T& i, T& stride) {
    i = blockIdx.x * blockDim.x + threadIdx.x + start;
    stride = blockDim.x * gridDim.x;
}
