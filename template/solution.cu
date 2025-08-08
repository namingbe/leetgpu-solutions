#include <common.cuh>
#include <iostream>

// TODO: Implement kernel
__global__ void kernel() {
    // Implementation here
}

int main(int argc, char** argv) {
    int size = (argc > 1) ? atoi(argv[1]) : 1024;
    
    // Allocate memory
    // TODO: Problem-specific allocation
    
    // Warmup
    kernel<<<1, 1>>>();
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    
    // Benchmark
    CudaTimer timer;
    const int iterations = 100;
    
    timer.startTimer();
    for (int i = 0; i < iterations; i++) {
        kernel<<<1, 1>>>();  // TODO: Proper launch config
    }
    float ms = timer.stopTimer();
    
    std::cout << "Average time: " << ms / iterations << " ms" << std::endl;
    
    // TODO: Validate results
    
    return 0;
}
