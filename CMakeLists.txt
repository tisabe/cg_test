cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 # -arch sm_30
    )

# The source files (no 'main')
set(src
      common.h
      global.h
      linalg_gpu.cu
      linalg_gpu.h
      linalg_cpu.cu
      linalg_cpu.h
    )

# The executable (has 'main')
cuda_add_executable(cg_test cg_test.cu ${src})
