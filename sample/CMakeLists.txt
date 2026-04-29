set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 寻找 OpenMP 依赖
find_package(OpenMP REQUIRED)

# 如果找到 OpenMP，全局添加它的编译标志
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(symqg_indexing symqg_indexing.cpp)
add_executable(symqg_querying symqg_querying.cpp)

add_executable(ivf_rabitq_indexing ivf_rabitq_indexing.cpp)
add_executable(ivf_rabitq_querying ivf_rabitq_querying.cpp)

add_executable(hnsw_rabitq_indexing hnsw_rabitq_indexing.cpp)
add_executable(hnsw_rabitq_querying hnsw_rabitq_querying.cpp)

add_executable(ivf_rabitq_querying_parallel ivf_rabitq_querying_parallel.cpp)

target_link_libraries(ivf_rabitq_querying_parallel OpenMP::OpenMP_CXX)
