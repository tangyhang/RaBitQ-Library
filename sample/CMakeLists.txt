cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
project(ivf_rabitq_demo LANGUAGES CXX CUDA)

# 直接指定为 V100 架构
set(CMAKE_CUDA_ARCHITECTURES "70")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mbmi2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -fno-stack-protector")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -mavx2,-mfma,-mbmi2")

# include_directories(BEFORE "/usr/local/cuda-12.6/include")

# 1. 查找系统的 CUDA 工具包 (专门为了获取 cuBLAS)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# ... 你的其他可执行文件保持不变 ...

# --- 原有的可执行文件 ---
add_executable(symqg_indexing symqg_indexing.cpp)
add_executable(symqg_querying symqg_querying.cpp)

add_executable(ivf_rabitq_indexing ivf_rabitq_indexing.cpp)
add_executable(ivf_rabitq_querying ivf_rabitq_querying.cpp)

add_executable(hnsw_rabitq_indexing hnsw_rabitq_indexing.cpp)
add_executable(hnsw_rabitq_querying hnsw_rabitq_querying.cpp)

# 2. 编译原生版本
add_executable(ivf_rabitq_raw_querying 
    ivf_rabitq_querying_raw.cpp 
    ivf_rabitq_raw.cu
)

add_executable(ivf_rabitq_raw_querying_cuvs 
    ivf_rabitq_querying_raw_cuvs.cpp 
    ivf_rabitq_raw_cuvs.cu
)


# 3. 仅仅链接 cuBLAS，没有任何 RAFT 依赖！
target_link_libraries(ivf_rabitq_raw_querying PRIVATE CUDA::cublas)
target_link_libraries(ivf_rabitq_raw_querying_cuvs PRIVATE CUDA::cublas)