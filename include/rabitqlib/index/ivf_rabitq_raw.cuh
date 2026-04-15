#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// // =========================================================
// // 【关键修复】：解决 GCC 11 与 NVCC 混合编译 AVX 头文件时的 Bug
// // NVCC 不认识这个 builtin，直接定义为空让它放行
// // =========================================================
// #if defined(__CUDACC__) && !defined(__builtin_ia32_serialize)
// #define __builtin_ia32_serialize()
// #endif
// // ---------------------------------------------------------
// // 核心魔法：破解 private 权限，让我们能直接读取底层的 ids_ 和 batch_data_
// // ---------------------------------------------------------
// #define private public
// #include "rabitqlib/index/ivf/ivf.hpp"
// #undef private

static constexpr int WARP_SIZE      = 32;
constexpr size_t FAST_SIZE = 32;

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16

// --- Tunables ---
using T    = float;
using IdxT = uint32_t;

using lut_dtype = __half;  // FP16 alternative

typedef uint32_t PID;

struct ClusterQueryPair {
  int cluster_idx;
  int query_idx;
};

// 3. 自定义 Thrust 排序规则 (按 cluster_idx 从小到大排)
struct PairSortCompare {
    __host__ __device__ 
    bool operator()(const ClusterQueryPair& a, const ClusterQueryPair& b) const {
        return a.cluster_idx < b.cluster_idx;
    }
};

class GPUClusterMeta {
   public:
    size_t num;          // Number of vectors in this cluster.
    size_t iter;         // Number of iterations for FastScan.
    size_t remain;       // Number of leftover vectors after blocks.
    size_t start_index;  // Combined offset: index of first vector in the flattened arrays.

    // Constructor: computes iter and REMAIN based on FAST_SIZE.
    __host__ __device__ GPUClusterMeta(size_t num, size_t start_idx)
      : num(num), start_index(start_idx)
    {
      iter   = num / FAST_SIZE;
      remain = num - iter * FAST_SIZE;
    }

    // default constructor (useful for vector initialize)
    GPUClusterMeta() : num(0), iter(0), remain(0), start_index(0) {}

    // Copy constructor.
    GPUClusterMeta(const GPUClusterMeta& other) = default;

    // Copy assignment.
    GPUClusterMeta& operator=(const GPUClusterMeta& other) = default;

    // Destructor.
    ~GPUClusterMeta() = default;
};


// =========================================================
// Bitwise Kernel 参数结构体
// =========================================================
struct ComputeBitwiseKernelParams {
    const ClusterQueryPair* d_sorted_pairs;
    const GPUClusterMeta* d_cluster_meta; // 注意替换为你实际的 Meta 结构体名
    
    // Query 相关
    const float* d_query;
    const uint32_t* d_packed_queries; // 4-bit 打包后的 Query 位平面
    const float* d_widths;            // Query 量化的缩放宽度
    const float* d_G_k1xSumq;         // Query 的预计算因子
    const float* d_centroid_distances;// 步骤1中算出的 Query 到 Centroid 距离
    const float* d_threshold; // 每个 Query 的当前 topk 距离阈值 (用于动态剪枝)
    
    // 底库相关
    const uint32_t* d_short_data;     // 1-bit 底库数据
    const float* d_short_factors;     // 底库的还原 Factors (SoA 布局)
    const uint32_t* d_pids;           // 原始 ID

    const float* d_G_kbxSumq;       // 新增
    const uint8_t* d_long_code;     // 新增
    const float* d_ex_factor;       // 新增
    size_t ex_bits;                 // 新增
    
    // 输出相关
    float* d_candidates_dists;
    uint32_t* d_candidates_pids;
    int* d_query_write_counters;      // 用于原子累加分配输出位置
    
    // 维度与控制参数
    int num_queries;
    int num_centroids;
    int D;                            // 向量维度 (Padded)
    int num_bits;                     // = 4
    int num_words;                    // D / 32
    int num_pairs;                    // nprobe * num_queries
    int max_candidates_per_query;     // 每个 Query 最多输出多少个候选 (通常是 nprobe * max_cluster_size)
    int nprobe;                        // 扫描多少个 Cluster
    int topk;                          // 最终返回多少个近邻
    int max_candidates_per_pair;     // max storage per pair, 1000 suggested
  };

// function to extract long codes
__device__ inline uint32_t extract_code(const uint8_t* codes, size_t d, size_t EX_BITS)
{
  size_t bitPos    = d * EX_BITS;
  size_t byteIdx   = bitPos >> 3;
  size_t bitOffset = bitPos & 7;
  uint32_t v       = codes[byteIdx] << 8;
  if (bitOffset + EX_BITS > 8) { v |= codes[byteIdx + 1]; }
  int shift = 16 - (bitOffset + EX_BITS);
  return (v >> shift) & ((1u << EX_BITS) - 1);
}

class IVF_RaBitQ_Raw {
private:
    cublasHandle_t cublas_handle;
    cudaStream_t stream;

    // rabitqlib::ivf::IVF cpu_ivf;

    float* d_centroids = nullptr;
    float* d_rotator = nullptr;     
    uint32_t* d_short_data = nullptr;
    float* d_short_factors = nullptr;
    uint8_t* d_long_codes = nullptr;   
    float* d_ex_factors = nullptr;

    uint32_t* d_ids = nullptr;
    GPUClusterMeta* d_cluster_meta = nullptr; // 存储每个 cluster 的 num_points 

    size_t num_vectors = 0;
    size_t dim = 0;
    size_t num_centroids = 0;
    size_t ex_bits = 0;
    size_t padded_dim = 0;

public:
    IVF_RaBitQ_Raw();
    ~IVF_RaBitQ_Raw();

    size_t max_cluster_size_ = 0;
    float best_rescaling_factor = 0.0f;

    void load_index(const std::string& index_file);

    // 接收原生 C 指针
    void load_from_raw_pointers(
        size_t num_vecs, size_t d, size_t p_d, size_t n_centroids, size_t ex_b,
        const float* h_centroids,
        const uint32_t* h_short_data, 
        const float* h_short_factors,
        const uint8_t* h_long_code,
        const float* h_ex_factors,
        const uint32_t* h_ids,
        const size_t* h_cluster_sizes
    );

    void search(const float* h_queries, 
                size_t num_queries, 
                size_t topk, 
                size_t nprobe, 
                float* h_final_dists, 
                uint32_t* h_final_pids); 
};