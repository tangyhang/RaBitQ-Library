#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cfloat> // 提供 FLT_MAX
#include <cub/cub.cuh>

// 引入头文件
#include "rabitqlib/index/ivf_rabitq_raw_cuvs.cuh"

// 辅助函数：带有严格边界保护的 extract_code
__device__ inline uint32_t extract_code(const uint8_t* ptr, uint32_t dim, uint32_t ex_bits, uint32_t max_bytes) {
    uint32_t bit_idx = dim * ex_bits;
    uint32_t byte_idx = bit_idx / 8;
    uint32_t bit_offset = bit_idx % 8;
    
    uint32_t val = ptr[byte_idx];
    // 防止跨字节读取时越出 long_code 的最大长度
    if (byte_idx + 1 < max_bytes) {
        val |= (ptr[byte_idx + 1] << 8);
    }
    return (val >> bit_offset) & ((1 << ex_bits) - 1);
}

// ==========================================
// 辅助宏：用于检查 CUDA 函数调用的返回值
// ==========================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==========================================
// Kernel: 计算范数及组合距离
// ==========================================
__global__ void row_norms_fused_kernel(const float* __restrict__ A, int A_rows, int A_cols,
                                       const float* __restrict__ B, int B_rows, int B_cols,
                                       float* __restrict__ A_norms, float* __restrict__ B_norms) {
  extern __shared__ float sdata[];
  const int tid     = threadIdx.x;
  const int lane    = tid & 31;
  const int warp_id = tid >> 5;
  const int nWarps  = (blockDim.x + 31) / 32;
  const int total_rows = A_rows + B_rows;

  for (int g = blockIdx.x; g < total_rows; g += gridDim.x) {
    const float* row_ptr; int cols; float* out_ptr;
    if (g < A_rows) {
      row_ptr = A + (size_t)g * A_cols; cols = A_cols; out_ptr = A_norms + g;
    } else {
      const int br = g - A_rows;
      row_ptr = B + (size_t)br * B_cols; cols = B_cols; out_ptr = B_norms + br;
    }
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
      float v = row_ptr[c]; sum += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) sdata[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
      float warp_sum = (tid < nWarps) ? sdata[tid] : 0.0f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) warp_sum += __shfl_down_sync(0xffffffff, warp_sum, off);
      if (tid == 0) *out_ptr = warp_sum;
    }
    __syncthreads();
  }
}

__global__ void add_norms_kernel(float* distances, const float* query_norms, const float* centroid_norms, int Q, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Q * K) {
    int q = idx / K; int k = idx % K;
    distances[idx] = distances[idx] + query_norms[q] + centroid_norms[k];
  }
}

// ==========================================
// Kernel: 预计算 Query Factors (修复 Bug 3)
// ==========================================
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__global__ void computeQueryFactorsWarpKernel(const T* d_query, float* d_G_k1xSumq, float* d_G_kbxSumq, size_t num_queries, size_t D, size_t ex_bits) {
  const int warp_id         = threadIdx.x / 32;
  const int lane_id         = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;
  const int query_idx       = blockIdx.x * warps_per_block + warp_id;
  
  if (query_idx >= num_queries) return;
  float c_1 = -static_cast<float>((1 << 1) - 1) / 2.0f;
  float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.0f;
  const T* query = d_query + query_idx * D;
  T sum = 0;
  for (int i = lane_id; i < D; i += 32) { sum += query[i]; }
  sum = warpReduceSum(sum);
  if (lane_id == 0) {
    d_G_k1xSumq[query_idx] = sum * c_1;
    d_G_kbxSumq[query_idx] = sum * c_b;
  }
}

template <typename T>
void computeQueryFactors(const T* d_query, float* d_G_k1xSumq, float* d_G_kbxSumq, size_t num_queries, size_t D, size_t ex_bits, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int warps_per_block   = threads_per_block / 32;
  const int blocks            = (num_queries + warps_per_block - 1) / warps_per_block;
  computeQueryFactorsWarpKernel<T><<<blocks, threads_per_block, 0, stream>>>(d_query, d_G_k1xSumq, d_G_kbxSumq, num_queries, D, ex_bits);
}

// ==========================================
// Kernel: 扁平化重排与 Top-K 聚合
// ==========================================
__global__ void flatten_pairs_kernel(const int* d_probe_idx, int num_queries, int nprobe, ClusterQueryPair* d_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * nprobe) {
        d_pairs[idx].cluster_idx = d_probe_idx[idx];
        d_pairs[idx].query_idx   = idx / nprobe;
    }
}

__global__ void select_topk_kernel(float* d_centroid_dists, int candidate_pool_size, int topk, float* d_final_vals, int* d_final_idx) {
    int query_id = blockIdx.x;
    int tid = threadIdx.x; 

    float* row_dists = d_centroid_dists + query_id * candidate_pool_size;
    float* out_vals  = d_final_vals + query_id * topk;
    int* out_idx     = d_final_idx + query_id * topk;

    __shared__ float s_vals[256];
    __shared__ int   s_idx[256];

    for (int k = 0; k < topk; k++) {
        float local_min = FLT_MAX;
        int local_idx = -1;

        for (int i = tid; i < candidate_pool_size; i += blockDim.x) {
            float val = row_dists[i];
            if (val < local_min) {
                local_min = val;
                local_idx = i;
            }
        }
        s_vals[tid] = local_min; s_idx[tid]  = local_idx;
        __syncthreads(); 

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_vals[tid + stride] < s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + stride];
                    s_idx[tid]  = s_idx[tid + stride];
                }
            }
            __syncthreads(); 
        }

        if (tid == 0) {
            out_vals[k] = s_vals[0];
            out_idx[k]  = s_idx[0];
            if (s_idx[0] != -1) { row_dists[s_idx[0]] = FLT_MAX; }
        }
        __syncthreads();
    }
}

// 修复 Bug 2: 将局部 Index 映射回真实的全局 PID
__global__ void map_pid_kernel(int* final_idx, uint32_t* candidate_pids, uint32_t* final_pids, int total_elements, int candidate_per_query, int topk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int q_id = idx / topk;
        int local_idx = final_idx[idx];
        if (local_idx != -1) {
            final_pids[idx] = candidate_pids[q_id * candidate_per_query + local_idx];
        }
    }
}

// ==========================================
// 类函数实现
// ==========================================
IVF_RaBitQ_Raw::IVF_RaBitQ_Raw() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
}

IVF_RaBitQ_Raw::~IVF_RaBitQ_Raw() {
    cublasDestroy(cublas_handle);
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_centroids)     CUDA_CHECK(cudaFree(d_centroids));
    if (d_rotator)       CUDA_CHECK(cudaFree(d_rotator));
    if (d_short_data)    CUDA_CHECK(cudaFree(d_short_data));
    if (d_short_factors) CUDA_CHECK(cudaFree(d_short_factors));
    if (d_long_codes)    CUDA_CHECK(cudaFree(d_long_codes));
    if (d_ex_factors)    CUDA_CHECK(cudaFree(d_ex_factors));
    if (d_ids)           CUDA_CHECK(cudaFree(d_ids));
    if (d_cluster_meta)  CUDA_CHECK(cudaFree(d_cluster_meta));
    free_workspace();
}

// 1. 简化的 Load 函数 (分配显存并拷贝重排好的数据)
void IVF_RaBitQ_Raw::load_from_raw_pointers(
        size_t num_vecs, size_t d, size_t p_d, size_t n_centroids, size_t ex_b,
        const float* h_centroids,
        const uint32_t* h_short_data, 
        const float* h_short_factors,
        const uint8_t* h_long_codes,
        const float* h_ex_factors,
        const uint32_t* h_ids,
        const size_t* h_cluster_sizes
) {
    this->num_vectors = num_vecs;
    this->dim = d;
    this->padded_dim = p_d;
    this->num_centroids = n_centroids;
    this->ex_bits = ex_b;

    // 构建 Cluster Meta
    std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
    size_t current_offset = 0;
    for (size_t i = 0; i < num_centroids; ++i) {
        h_cluster_meta[i] = GPUClusterMeta(h_cluster_sizes[i], current_offset);
        current_offset += h_cluster_sizes[i];
    }
    CUDA_CHECK(cudaMallocAsync(&d_cluster_meta, num_centroids * sizeof(GPUClusterMeta), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cluster_meta, h_cluster_meta.data(), num_centroids * sizeof(GPUClusterMeta), cudaMemcpyHostToDevice, stream));

    // 上传 Centroids
    size_t centroids_bytes = num_centroids * padded_dim * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_centroids, centroids_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice, stream));

    // 上传我们自己重排好的 1-bit Codes 和 Factors
    size_t short_data_bytes = num_vectors * (padded_dim / 32) * sizeof(uint32_t);
    CUDA_CHECK(cudaMallocAsync(&d_short_data, short_data_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_short_data, h_short_data, short_data_bytes, cudaMemcpyHostToDevice, stream));

    size_t short_factors_bytes = num_vectors * 3 * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_short_factors, short_factors_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_short_factors, h_short_factors, short_factors_bytes, cudaMemcpyHostToDevice, stream));

    // [新增] 上传 Extended Data
    if (ex_b > 0) {
        size_t long_code_bytes = num_vectors * ((padded_dim * ex_b + 7) / 8);
        CUDA_CHECK(cudaMallocAsync(&d_long_codes, long_code_bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_long_codes, h_long_codes, long_code_bytes, cudaMemcpyHostToDevice, stream));

        size_t ex_factors_bytes = num_vectors * 2 * sizeof(float);
        CUDA_CHECK(cudaMallocAsync(&d_ex_factors, ex_factors_bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ex_factors, h_ex_factors, ex_factors_bytes, cudaMemcpyHostToDevice, stream));
    }

    // 上传 PIDs
    size_t ids_bytes = num_vectors * sizeof(uint32_t);
    CUDA_CHECK(cudaMallocAsync(&d_ids, ids_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ids, h_ids, ids_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "Engine loaded via Custom Manual Transpose successfully!" << std::endl;
}



void IVF_RaBitQ_Raw::prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe) {
    free_workspace(); // 如果已经分配过，先清空

    ws_max_queries = max_queries;
    ws_max_topk = max_topk;
    ws_max_nprobe = max_nprobe;

    size_t max_total_probes = max_queries * max_nprobe;
    // size_t max_total_candidates = max_total_probes * this->max_cluster_size_;
    size_t max_total_candidates = max_queries * max_nprobe * max_topk;
    // 在 prepare_workspace 中加入：
    CUDA_CHECK(cudaMallocAsync(&ws_d_threshold, max_queries * sizeof(float), stream));

    // 一次性分配所有可能会在 Search 中用到的临时显存
    CUDA_CHECK(cudaMallocAsync(&ws_d_queries, max_queries * padded_dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_dists, max_queries * num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_dists_copy, max_queries * num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_q_norms, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_c_norms, num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_vals, max_queries * max_nprobe * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_idx, max_queries * max_nprobe * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_sorted_pairs, max_total_probes * sizeof(ClusterQueryPair), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_k1xSumq, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_kbxSumq, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_dists, max_total_candidates * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_pids, max_total_candidates * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_query_write_counters, max_queries * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_final_dev_dists, max_queries * max_topk * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_final_dev_pids, max_queries * max_topk * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void IVF_RaBitQ_Raw::free_workspace() {
    if(ws_d_queries) cudaFree(ws_d_queries);
    if(ws_d_centroid_dists) cudaFree(ws_d_centroid_dists);
    if(ws_d_centroid_dists_copy) cudaFree(ws_d_centroid_dists_copy);
    if(ws_d_q_norms) cudaFree(ws_d_q_norms);
    if(ws_d_c_norms) cudaFree(ws_d_c_norms);
    if(ws_d_probe_vals) cudaFree(ws_d_probe_vals);
    if(ws_d_probe_idx) cudaFree(ws_d_probe_idx);
    if(ws_d_sorted_pairs) cudaFree(ws_d_sorted_pairs);
    if(ws_d_G_k1xSumq) cudaFree(ws_d_G_k1xSumq);
    if(ws_d_G_kbxSumq) cudaFree(ws_d_G_kbxSumq);
    if(ws_d_candidates_dists) cudaFree(ws_d_candidates_dists);
    if(ws_d_candidates_pids) cudaFree(ws_d_candidates_pids);
    if(ws_d_query_write_counters) cudaFree(ws_d_query_write_counters);
    if(ws_d_final_dev_dists) cudaFree(ws_d_final_dev_dists);
    if(ws_d_final_dev_pids) cudaFree(ws_d_final_dev_pids);

    // 在 free_workspace 中加入：
    if(ws_d_threshold) cudaFree(ws_d_threshold); ws_d_threshold = nullptr;

    ws_d_queries = nullptr; ws_d_centroid_dists = nullptr; ws_d_centroid_dists_copy = nullptr;
    ws_d_q_norms = nullptr; ws_d_c_norms = nullptr; ws_d_probe_vals = nullptr;
    ws_d_probe_idx = nullptr; ws_d_sorted_pairs = nullptr; ws_d_G_k1xSumq = nullptr;
    ws_d_G_kbxSumq = nullptr; ws_d_candidates_dists = nullptr; ws_d_candidates_pids = nullptr;
    ws_d_query_write_counters = nullptr; ws_d_final_dev_dists = nullptr; ws_d_final_dev_pids = nullptr;
}

__global__ void computeInnerProductsWithBitwiseOpt(const ComputeBitwiseKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Allocate shared memory for candidates
  float* shared_query       = reinterpret_cast<float*>(shared_mem_raw_2);
  float* shared_ip2_results = shared_query + params.D;

  for (size_t i = tid; i < params.D; i += num_threads) {
    shared_query[i] = params.d_query[query_idx * params.D + i];
  }
  __syncthreads();

  // Step 1: Warp-level IP2 computation for better memory coalescing
  const int warp_id   = tid / WARP_SIZE;
  const int lane_id   = tid % WARP_SIZE;
  const int num_warps = num_threads / WARP_SIZE;

  // Calculate long code parameters
  const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

  // Each warp processes different candidates
  for (int cand_idx = warp_id; cand_idx < num_vectors_in_cluster; cand_idx += num_warps) {
    size_t global_vec_idx = cluster_start_index + cand_idx;

    // Pointer to this vector's long code
    const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

    // Warp-level IP2 computation
    float ip2 = 0.0f;

    // 为了调试，我们开一个临时变量记录该线程处理的维度点乘和
    float debug_thread_ip2_sum = 0.0f;

    // Each thread in warp processes different dimensions
    for (uint32_t d = lane_id; d < params.D; d += WARP_SIZE) {
      uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits, long_code_size);

      float ex_val      = (float)code_val;
      ip2 += shared_query[d] * ex_val;
    }

    // Warp-level reduction for ip2
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
    }

    // Lane 0 stores the result
    if (lane_id == 0) { 
      shared_ip2_results[cand_idx] = ip2; 
    }
  }

  __syncthreads();

  // Step 2: Load float query and compute exact IPs for candidates

  const size_t short_code_length = params.D / 32;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_kbxsumq = params.d_G_kbxSumq[query_idx];

  // Atomically get write position
  __shared__ int probe_slot;
  if (tid == 0) {
    probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
  }
  __syncthreads();
  // Calculate output offset
  uint32_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx >= num_vectors_in_cluster) break;

    // Compute exact inner product with float query
    float exact_ip = 0.0f;

    // Process each uint32_t of the short code
    for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
      // Access short code in transposed layout
      size_t short_code_offset =
        cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
      uint32_t short_code_chunk = params.d_short_data[short_code_offset];

      // Process each bit in the uint32_t
      // Note: bit 31 is lowest dimension, bit 0 is highest
#pragma unroll 8
      for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
        size_t dim = uint32_idx * 32 + bit_idx;
        if (dim < params.D) {
          // Extract bit from MSB to LSB
          int bit_position = 31 - bit_idx;
          bool bit_value   = (short_code_chunk >> bit_position) & 0x1;

          // If bit is 1, add the query value
          if (bit_value) { 
            exact_ip += shared_query[dim]; 
          } 
        }
      }
    }


    // Get pre-computed values
    float ip2             = shared_ip2_results[vec_idx];
    size_t global_vec_idx = cluster_start_index + vec_idx;

    // vec load version
    float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
    float f_ex_add     = ex_factors.x;
    float f_ex_rescale = ex_factors.y;

    // Compute final distance using pre-computed ip2
    float ex_dist =
      f_ex_add + q_g_add +
      f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);

    // Write to global memory
    params.d_candidates_dists[output_offset + vec_idx] = ex_dist;
    params.d_candidates_pids[output_offset + vec_idx]  = params.d_pids[global_vec_idx];
    
}
}

// 🌟 安全处理所有浮点数的原子求小操作 (保留)
__device__ __forceinline__ void atomicMinFloat(const float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__global__ void computeInnerProductsWithBitwiseOptExactPruning(const ComputeBitwiseKernelParams params) {
    int block_id = blockIdx.x;
    if (block_id >= params.num_pairs) return;

    ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx = pair.query_idx;
    size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
    size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // =========================================================
    // 🌟 计时变量定义 (仅在 Block 0, Thread 0 记录)
    // =========================================================
    long long t_start = 0, t_phase1 = 0, t_phase2 = 0, t_phase3 = 0;

    // --- 动态 Shared Memory 布局 ---
    extern __shared__ char smem[];
    float* s_query = (float*)smem;
    float* s_exact_ip = s_query + params.D;
    int* s_passed_c = (int*)(s_exact_ip + params.max_candidates_per_pair);
    float* s_cand_ips = (float*)(s_passed_c + params.max_candidates_per_pair);
    int* s_cand_idx = (int*)(s_cand_ips + params.max_candidates_per_pair);

    // 加载 Query
    for (int i = tid; i < params.D; i += num_threads) {
        s_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    // 🌟 记录起点：屏蔽了加载 Query 的开销，直接测算核心逻辑
    if (block_id == 0 && tid == 0) t_start = clock64();

    __shared__ int num_passed;
    if (tid == 0) num_passed = 0;
    __syncthreads();

    float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    float q_k1xsumq = params.d_G_k1xSumq[query_idx];
    float q_kbxsumq = params.d_G_kbxSumq[query_idx];
    float q_g_error = sqrtf(q_g_add);
    float threshold = params.d_threshold[query_idx];

    int actual_cands = min((int)num_vectors_in_cluster, params.max_candidates_per_pair);
    size_t short_code_length = params.D / 32;

    // -------------------------------------------------------------
    // 阶段 1：无分支算 1-bit Exact IP + 精确初筛
    // -------------------------------------------------------------
    for (int c = tid; c < actual_cands; c += num_threads) {
        size_t global_idx = cluster_start_index + c;
        float exact_ip = 0.0f;

        // 【最快】无分支浮点相乘，暴力碾压
        for (int w = 0; w < short_code_length; w++) {
            uint32_t chunk = params.d_short_data[cluster_start_index * short_code_length + w * num_vectors_in_cluster + c];
            #pragma unroll 8
            for (int b = 0; b < 32; b++) {
                int dim = w * 32 + b;
                if (dim < params.D) {
                    float bit_val = (float)((chunk >> (31 - b)) & 0x1);
                    exact_ip += s_query[dim] * bit_val;
                }
            }
        }
        
        s_exact_ip[c] = exact_ip; 

        float3 factors = reinterpret_cast<const float3*>(params.d_short_factors)[global_idx];
        
        float low_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;
        
        if (low_dist < threshold) {
            int slot = atomicAdd(&num_passed, 1);
            if (slot < params.max_candidates_per_pair) {
                s_passed_c[slot] = c;
            }
        }
    }
    __syncthreads();

    // 🌟 记录阶段 1 结束
    if (block_id == 0 && tid == 0) t_phase1 = clock64();

    // 全局写入偏移
    __shared__ int probe_slot;
    if (tid == 0) probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1);
    __syncthreads();
    if (probe_slot >= params.nprobe) return;
    uint32_t out_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;

    int passed = min(num_passed, params.max_candidates_per_pair);

    // -------------------------------------------------------------
    // 阶段 2：Warp 并行算长码 IP2 (仅针对通过初筛的少数精锐)
    // -------------------------------------------------------------
    int warp_id = tid / 32, lane_id = tid % 32, num_warps = num_threads / 32;
    uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

    if (passed > 0) {
        for (int idx = warp_id; idx < passed; idx += num_warps) {
            int c = s_passed_c[idx];
            size_t global_idx = cluster_start_index + c;
            
            float ip2 = 0.0f;
            const uint8_t* vec_long_code = params.d_long_code + global_idx * long_code_size;
            
            for (int d = lane_id; d < params.D; d += 32) {
                uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits, long_code_size);
                ip2 += s_query[d] * (float)code_val;
            }
            
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
            
            if (lane_id == 0) {
                float exact_ip = s_exact_ip[c];
                float2 ex_factors = reinterpret_cast<const float2*>(params.d_ex_factor)[global_idx];
                float ex_dist = ex_factors.x + q_g_add + ex_factors.y * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);
                
                s_cand_ips[idx] = ex_dist;
                s_cand_idx[idx] = params.d_pids[global_idx];
            }
        }
    }
    __syncthreads();

    // 🌟 记录阶段 2 结束
    if (block_id == 0 && tid == 0) t_phase2 = clock64();

    // -------------------------------------------------------------
    // 阶段 3 & 4：K-pass 极速规约 BlockSort & 更新阈值
    // -------------------------------------------------------------
    float* s_min_val = s_exact_ip; // 安全复用
    int* s_min_idx = (int*)(s_min_val + blockDim.x);

    if (passed > 0) {
        for (int k = 0; k < params.topk; k++) {
            float local_min = FLT_MAX;
            int local_idx = -1;

            for (int i = tid; i < passed; i += num_threads) {
                float val = s_cand_ips[i];
                if (val < local_min) { local_min = val; local_idx = i; }
            }
            s_min_val[tid] = local_min; s_min_idx[tid] = local_idx;
            __syncthreads();

            for (int stride = 128; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    if (s_min_val[tid + stride] < s_min_val[tid]) {
                        s_min_val[tid] = s_min_val[tid + stride];
                        s_min_idx[tid] = s_min_idx[tid + stride];
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                if (s_min_idx[0] != -1) {
                    int best_idx = s_min_idx[0];
                    params.d_candidates_dists[out_offset + k] = s_cand_ips[best_idx];
                    params.d_candidates_pids[out_offset + k]  = s_cand_idx[best_idx];
                    s_cand_ips[best_idx] = FLT_MAX; 
                } else {
                    params.d_candidates_dists[out_offset + k] = FLT_MAX;
                    params.d_candidates_pids[out_offset + k]  = 0;
                }
            }
            __syncthreads();
        }

        if (passed >= params.topk && tid == 0) {
            float max_topk_dist = params.d_candidates_dists[out_offset + params.topk - 1];
            if (max_topk_dist < threshold) {
                atomicMinFloat(params.d_threshold + query_idx, max_topk_dist);
            }
        }
    } else {
        if (tid == 0) {
            for (int k = 0; k < params.topk; k++) {
                params.d_candidates_dists[out_offset + k] = FLT_MAX;
                params.d_candidates_pids[out_offset + k]  = 0;
            }
        }
    }

    // 🌟 记录阶段 3 & 4 结束并打印
    if (block_id == 0 && tid == 0) {
        t_phase3 = clock64();
        printf("\n========== [Kernel Internal Profiler (Block 0)] ==========\n");
        printf("  ├─ 1. Phase 1 (1-bit IP + Prune) : %lld cycles\n", t_phase1 - t_start);
        printf("  ├─ 2. Phase 2 (Long Code IP2)    : %lld cycles\n", t_phase2 - t_phase1);
        printf("  └─ 3. Phase 3 & 4 (Sort & Atomic): %lld cycles\n", t_phase3 - t_phase2);
        printf("==========================================================\n");
    }
}

void IVF_RaBitQ_Raw::search(const float* h_queries, size_t num_queries, size_t topk, size_t nprobe, float* h_final_dists, uint32_t* h_final_pids) {
    // -------------------------------------------------------------
    // 安全检查：确保当前查询的规模没有超出我们预分配的 Workspace 容量
    // -------------------------------------------------------------
    if (num_queries > ws_max_queries || topk > ws_max_topk || nprobe > ws_max_nprobe) {
        std::cerr << "[Error] Search parameters exceed workspace capacity!" << std::endl;
        return;
    }

    // 将预分配好的显存指针映射到局部变量，避免重写后续业务代码
    float* d_queries              = ws_d_queries;
    float* d_centroid_dists       = ws_d_centroid_dists;
    float* d_centroid_dists_copy  = ws_d_centroid_dists_copy;
    float* d_q_norms              = ws_d_q_norms;
    float* d_c_norms              = ws_d_c_norms;
    float* d_probe_vals           = ws_d_probe_vals;
    int* d_probe_idx            = ws_d_probe_idx;
    ClusterQueryPair* d_sorted_pairs = ws_d_sorted_pairs;
    float* d_G_k1xSumq            = ws_d_G_k1xSumq;
    float* d_G_kbxSumq            = ws_d_G_kbxSumq;
    float* d_candidates_dists     = ws_d_candidates_dists;
    uint32_t* d_candidates_pids   = ws_d_candidates_pids;
    int* d_query_write_counters   = ws_d_query_write_counters;
    float* d_final_dev_dists      = ws_d_final_dev_dists;
    uint32_t* d_final_dev_pids    = ws_d_final_dev_pids;
    
    // =========================================================
    // 🌟 初始化 CUDA 事件，用于高精度测速
    // =========================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // 定义测速宏，保持代码整洁
    #define TICK() cudaEventRecord(start, stream)
    #define TOCK(step_name) \
        cudaEventRecord(stop, stream); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&ms, start, stop); \
        std::cout << "[Profile] " << step_name << " 耗时: " << ms << " ms\n"


    // =========================================================
    // 0. Query 数据拷贝 (H2D)
    // =========================================================
    TICK();
    size_t queries_bytes = num_queries * padded_dim * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_queries, h_queries, queries_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("0. Query H2D Transfer");

    // =========================================================
    // 1. 粗量化检索 (CUBLAS GEMM)
    // 利用矩阵乘法快速计算 Query 与所有簇中心 (Centroids) 的内积
    // =========================================================
    TICK();
    const float alpha = -2.0f; const float beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                num_centroids, num_queries, padded_dim,
                &alpha, d_centroids, padded_dim, 
                d_queries, padded_dim, 
                &beta, d_centroid_dists, num_centroids);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("1. Coarse Search (cuBLAS)");

    // =========================================================
    // 2. 补全 L2 距离范数
    // 加上 Query 自身范数和 Centroid 范数，构成完整的 L2 距离
    // =========================================================
    TICK();
    int norm_block_size = 256;
    size_t norm_shared_mem = ((norm_block_size + 31) / 32) * sizeof(float);
    row_norms_fused_kernel<<<num_centroids + num_queries, norm_block_size, norm_shared_mem, stream>>>(
        d_queries, num_queries, padded_dim, d_centroids, num_centroids, padded_dim, d_q_norms, d_c_norms);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int add_threads = 256;
    int add_blocks  = (num_queries * num_centroids + add_threads - 1) / add_threads;
    add_norms_kernel<<<add_blocks, add_threads, 0, stream>>>(d_centroid_dists, d_q_norms, d_c_norms, num_queries, num_centroids);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("2. L2 Norms Completion");

    // =========================================================
    // 3. Top-nprobe 簇选择
    // 为每个 Query 选出距离最近的 nprobe 个簇，准备进行细排
    // =========================================================
    TICK();
    // 拷贝一份用于破坏性 TopK 筛选
    CUDA_CHECK(cudaMemcpyAsync(d_centroid_dists_copy, d_centroid_dists, num_queries * num_centroids * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    select_topk_kernel<<<num_queries, 256, 0, stream>>>(d_centroid_dists_copy, num_centroids, nprobe, d_probe_vals, d_probe_idx);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("3. Select Top-nprobe Clusters");

    // =========================================================
    // 4. Pair 扁平化与聚类排序
    // 将待查的簇转为 <Cluster, Query> 键值对，并按 Cluster 排序以增加访存局部性
    // =========================================================
    TICK();
    int total_probes = num_queries * nprobe;
    flatten_pairs_kernel<<<(total_probes + 255) / 256, 256, 0, stream>>>(d_probe_idx, num_queries, nprobe, d_sorted_pairs);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::sort(thrust::cuda::par.on(stream), d_sorted_pairs, d_sorted_pairs + total_probes, PairSortCompare());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("4. Flatten & Sort Pairs");

    // =========================================================
    // 5. 预计算 Query 还原因子
    // 提前算出 RaBitQ 公式中与 Query 相关的常数项
    // =========================================================
    TICK();
    computeQueryFactors<float>(d_queries, d_G_k1xSumq, d_G_kbxSumq, num_queries, padded_dim, ex_bits, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("5. Precompute Query Factors");

    // -------------------------------------------------------------
    // 初始化动态阈值为无穷大 (让第一个 Block 畅通无阻，建立基准)
    // -------------------------------------------------------------
    TICK();
    thrust::fill(thrust::cuda::par.on(stream), thrust::device_pointer_cast(ws_d_threshold), thrust::device_pointer_cast(ws_d_threshold) + num_queries, FLT_MAX);
    CUDA_CHECK(cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream));
    TOCK("6. Candidate Threshold & Counters Init");

    // -------------------------------------------------------------
    // 7. 终极细粒度检索 (动态剪枝 + __clz 无分支 + BlockSort)
    // -------------------------------------------------------------
    TICK();
    int num_words = (padded_dim + 31) / 32;

    ComputeBitwiseKernelParams bitwiseParams;
    bitwiseParams.d_sorted_pairs = d_sorted_pairs;
    bitwiseParams.d_cluster_meta = d_cluster_meta;
    bitwiseParams.d_query = d_queries; 
    bitwiseParams.d_packed_queries = nullptr; // 无需量化
    bitwiseParams.d_widths = nullptr;         // 无需量化
    bitwiseParams.d_threshold = ws_d_threshold; // 🌟 传入剪枝阈值！
    bitwiseParams.d_G_k1xSumq = d_G_k1xSumq; 
    bitwiseParams.d_centroid_distances = d_centroid_dists;
    bitwiseParams.d_short_data = d_short_data;
    bitwiseParams.d_short_factors = d_short_factors;
    bitwiseParams.d_long_code = d_long_codes;         
    bitwiseParams.d_ex_factor = d_ex_factors;         
    bitwiseParams.d_G_kbxSumq = d_G_kbxSumq;
    bitwiseParams.ex_bits = ex_bits;
    bitwiseParams.d_pids = d_ids;
    bitwiseParams.d_candidates_dists = d_candidates_dists;
    bitwiseParams.d_candidates_pids = d_candidates_pids;
    bitwiseParams.d_query_write_counters = d_query_write_counters;
    bitwiseParams.num_queries = num_queries;
    bitwiseParams.num_centroids = num_centroids;
    bitwiseParams.D = padded_dim;
    bitwiseParams.num_words = num_words;
    bitwiseParams.num_pairs = total_probes;
    bitwiseParams.max_candidates_per_pair = this->max_cluster_size_;
    bitwiseParams.topk = topk;
    bitwiseParams.nprobe = nprobe;

    // 共享内存分配：确保足够容纳所有中间状态
    size_t query_bytes = padded_dim * sizeof(float);
    size_t cand_bytes = this->max_cluster_size_ * (3 * sizeof(float) + 2 * sizeof(int));
    size_t reduce_bytes = 256 * (sizeof(float) + sizeof(int));
    size_t shared_mem_size = query_bytes + cand_bytes + reduce_bytes;

    computeInnerProductsWithBitwiseOptExactPruning<<<total_probes, 256, shared_mem_size, stream>>>(bitwiseParams);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("7. Fine Search (Exact Dynamic Pruning)");
    
    // -------------------------------------------------------------
    // 8. 全局聚合 
    // -------------------------------------------------------------
    TICK();
    size_t final_bytes = num_queries * topk * sizeof(float);
    size_t reduced_pool_size = nprobe * topk; 
    
    select_topk_kernel<<<num_queries, 256, 0, stream>>>(d_candidates_dists, reduced_pool_size, topk, d_final_dev_dists, (int*)d_final_dev_pids);
    map_pid_kernel<<<(num_queries * topk + 255) / 256, 256, 0, stream>>>((int*)d_final_dev_pids, d_candidates_pids, d_final_dev_pids, num_queries * topk, reduced_pool_size, topk);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("8. Global Top-K & PID Mapping");

    // // =========================================================
    // // 7. 细粒度精确检索 (Bitwise Inner Product)
    // // 这是核心算子：读取 1-bit 短码与长码进行距离融合计算
    // // =========================================================
    // TICK();
    // int num_words = (padded_dim + 31) / 32;
    
    // ComputeBitwiseKernelParams bitwiseParams;
    // bitwiseParams.d_sorted_pairs = d_sorted_pairs;
    // bitwiseParams.d_cluster_meta = d_cluster_meta;
    // bitwiseParams.d_query = d_queries; 
    // bitwiseParams.d_packed_queries = nullptr; 
    // bitwiseParams.d_widths = nullptr;         
    // bitwiseParams.d_G_k1xSumq = d_G_k1xSumq; 
    // bitwiseParams.d_centroid_distances = d_centroid_dists;
    // bitwiseParams.d_short_data = d_short_data;
    // bitwiseParams.d_short_factors = d_short_factors;
    // bitwiseParams.d_long_code = d_long_codes;         
    // bitwiseParams.d_ex_factor = d_ex_factors;         
    // bitwiseParams.d_G_kbxSumq = d_G_kbxSumq;
    // bitwiseParams.ex_bits = ex_bits;
    // bitwiseParams.d_pids = d_ids;
    // bitwiseParams.d_candidates_dists = d_candidates_dists;
    // bitwiseParams.d_candidates_pids = d_candidates_pids;
    // bitwiseParams.d_query_write_counters = d_query_write_counters;
    // bitwiseParams.num_queries = num_queries;
    // bitwiseParams.num_centroids = num_centroids;
    // bitwiseParams.D = padded_dim;
    // bitwiseParams.num_bits = 4; 
    // bitwiseParams.num_words = num_words;
    // bitwiseParams.num_pairs = total_probes;
    // bitwiseParams.max_candidates_per_query = pool_size_per_query;

    // size_t shared_mem_size = (padded_dim + this->max_cluster_size_) * sizeof(float);
    // computeInnerProductsWithBitwiseOpt<<<total_probes, 256, shared_mem_size, stream>>>(bitwiseParams);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // TOCK("7. Fine Search (Bitwise Kernel)");
    
    // // =========================================================
    // // 8. 全局最终 Top-K 聚合与 PID 映射
    // // 从候选池中筛选出每个 Query 的最终 Top-K 并转为真实全局 ID
    // // =========================================================
    // TICK();
    // size_t final_bytes = num_queries * topk * sizeof(float);
    // select_topk_kernel<<<num_queries, 256, 0, stream>>>(d_candidates_dists, pool_size_per_query, topk, d_final_dev_dists, (int*)d_final_dev_pids);
    // map_pid_kernel<<<(num_queries * topk + 255) / 256, 256, 0, stream>>>((int*)d_final_dev_pids, d_candidates_pids, d_final_dev_pids, num_queries * topk, pool_size_per_query, topk);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // TOCK("8. Global Top-K & PID Mapping");

    // =========================================================
    // 9. 结果拷贝回 Host (D2H) & 清理状态
    // =========================================================
    TICK();
    CUDA_CHECK(cudaMemcpyAsync(h_final_dists, d_final_dev_dists, final_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_final_pids, d_final_dev_pids, num_queries * topk * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //打印查询0的h_final_dists和h_final_pids的前10个元素以验证结果
    // for (int i = 0; i < std::min(10, (int)(num_queries * topk)); i++) {
    //     if (i < topk) {
    //         std::cout << "Query 0, Rank " << i << ": Dist = " << h_final_dists[i] << ", PID = " << h_final_pids[i] << "\n";
    //     }
    // }
    CUDA_CHECK(cudaMemsetAsync(d_final_dev_pids, 0, num_queries * topk * sizeof(uint32_t), stream));
    TOCK("9. Results D2H Transfer");

    std::cout << "------------------------------------------\n";

    // 释放测速组件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}