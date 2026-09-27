cmake_minimum_required(VERSION 3.18)
project(CliffordLTC_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Architecture flags: Support Turing (sm_75 like GTX 1660 Ti), Ampere (sm_80/86), Ada (sm_89), Hopper (sm_90)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Compiler options
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math -Xcompiler -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Clifford-LTC CUDA Library
add_library(clifford_cuda STATIC
    src/clifford_fwd_kernel.cu
    src/clifford_bwd_kernel.cu
)

set_target_properties(clifford_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Test & Benchmark Executable
add_executable(test_clifford_cuda
    tests/test_clifford_cuda.cu
)

target_link_libraries(test_clifford_cuda PRIVATE clifford_cuda)
