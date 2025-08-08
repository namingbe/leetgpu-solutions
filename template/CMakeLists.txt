get_filename_component(PROBLEM_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable(${PROBLEM_NAME} solution.cu)
target_link_libraries(${PROBLEM_NAME} cudnn cublas)

# Enable for profiling
set_target_properties(${PROBLEM_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
