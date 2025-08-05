cmake_minimum_required(VERSION 3.18)
project(leetgpu_solutions LANGUAGES CUDA CXX)

include(kitchen/cmake/CUDAConfig.cmake)

# Build all solutions
file(GLOB PROBLEM_DIRS */CMakeLists.txt)
foreach(problem_cmake ${PROBLEM_DIRS})
    get_filename_component(problem_dir ${problem_cmake} DIRECTORY)
    get_filename_component(problem_name ${problem_dir} NAME)
    
    # Skip kitchen and template directories
    if(NOT ${problem_name} MATCHES "^(kitchen|template)$")
        add_subdirectory(${problem_dir})
    endif()
endforeach()
