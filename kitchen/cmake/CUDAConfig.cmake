# Auto-detect GPU architecture
include(FindCUDA/select_compute_arch)
cuda_detect_installed_gpus(DETECTED_ARCHS)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Performance flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3 -Xcompiler -march=native")

# Verbose PTX to see register/shared memory usage
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

# Debug build
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Include common headers
include_directories(${CMAKE_SOURCE_DIR}/kitchen/include)
