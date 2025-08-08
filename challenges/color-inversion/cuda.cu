// CUDA 12.8.0
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < height * width * 4 && x % 4 != 3) {
        image[x] = 255 - image[x];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height * 4 + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
