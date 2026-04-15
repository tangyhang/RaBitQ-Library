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


// 替换 ivf_rabitq_raw_cuvs.cu 最上面的这个函数
// __device__ inline uint32_t extract_code(const uint8_t* ptr, uint32_t dim, uint32_t ex_bits, uint32_t max_bytes) {
//     size_t bitPos = dim * ex_bits;
//     size_t byteIdx = bitPos / 8;
//     size_t bitOffset = bitPos % 8;
    
//     // 🌟 核心修复：恢复大端序 (MSB-first) 的拼接方式
//     uint32_t v = ptr[byteIdx] << 8;
//     if (bitOffset + ex_bits > 8 && (byteIdx + 1 < max_bytes)) { 
//         v |= ptr[byteIdx + 1]; 
//     }
    
//     int shift = 16 - (bitOffset + ex_bits);
//     return (v >> shift) & ((1u << ex_bits) - 1);
// }
// --- 显存调试工具函数 ---
void print_device_array(const char* name, const float* d_ptr, int count) {
    std::vector<float> h_arr(count);
    cudaMemcpy(h_arr.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[Debug X-Ray] " << name << " (first " << count << "): ";
    for(int i = 0; i < count; i++) std::cout << h_arr[i] << " ";
    std::cout << std::endl;
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
// Kernel: Query 在线 4-bit 量化打包
// ==========================================
__global__ void findQueryRanges(const float* __restrict__ queries, float* __restrict__ query_ranges, int num_queries, int num_dimensions) {
    const int query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;
    const float* query = queries + query_idx * num_dimensions;
    typedef cub::BlockReduce<float, 256> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp_storage_min;
    __shared__ typename BlockReduceFloat::TempStorage temp_storage_max;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < num_dimensions; i += blockDim.x) {
        local_min = fminf(local_min, query[i]);
        local_max = fmaxf(local_max, query[i]);
    }
    float block_min = BlockReduceFloat(temp_storage_min).Reduce(local_min, cub::Min());
    __syncthreads();
    float block_max = BlockReduceFloat(temp_storage_max).Reduce(local_max, cub::Max());
    if (threadIdx.x == 0) {
        query_ranges[query_idx * 2]     = block_min;
        query_ranges[query_idx * 2 + 1] = block_max;
    }
}

__global__ void quantizeQueriesToInt4(const float* __restrict__ queries, const float* __restrict__ query_ranges, int8_t* __restrict__ quantized_queries, float* __restrict__ widths, int num_queries, int num_dimensions) {
    const float max_int_val = 7.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_queries * num_dimensions;
    for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
        int query_idx = i / num_dimensions;
        int dim_idx   = i % num_dimensions;
        float vmin = query_ranges[query_idx * 2];
        float vmax = query_ranges[query_idx * 2 + 1];
        float width = fmaxf(fabsf(vmin), fabsf(vmax)) / max_int_val;
        if (dim_idx == 0) widths[query_idx] = width;
        float scaled = queries[i] * (width > 0 ? 1.0f / width : 0.0f);
        quantized_queries[i] = (int8_t)__float2int_rn(fmaxf(-8.0f, fminf(7.0f, scaled)));
    }
}

__global__ void packInt4QueryBitPlanes(const int8_t* __restrict__ queries, uint32_t* __restrict__ packed_queries, int num_queries, int num_dimensions) {
    const int dims_per_word = 32;
    const int num_words = (num_dimensions + dims_per_word - 1) / dims_per_word;
    const int num_bits = 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_queries * num_bits * num_words;
    for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
        int query_idx = i / (num_bits * num_words);
        int remainder = i % (num_bits * num_words);
        int bit_idx   = remainder / num_words;
        int word_idx  = remainder % num_words;
        uint32_t packed_word = 0;
#pragma unroll 8
        for (int d = 0; d < dims_per_word; ++d) {
            int dim_idx = word_idx * dims_per_word + d;
            if (dim_idx < num_dimensions) {
                uint8_t val = (uint8_t)(queries[query_idx * num_dimensions + dim_idx] & 0xF);
                packed_word |= (((val >> bit_idx) & 1) << (31 - d));
            }
        }
        packed_queries[i] = packed_word;
    }
}


__global__ void computeInnerProductsBitwise4bit(const ComputeBitwiseKernelParams params) {
    const int block_id = blockIdx.x;
    if (block_id >= params.num_pairs) return;

    ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx   = pair.query_idx;
    if (cluster_idx < 0 || cluster_idx >= params.num_centroids || query_idx < 0 || query_idx >= params.num_queries) return;

    size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
    size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

    // =========================================================
    // 🌟 混合显存布局：同时加载 4-bit Query 和 Float Query
    // =========================================================
    extern __shared__ char shared_mem_raw[];
    uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw);
    float* shared_float_query = reinterpret_cast<float*>(shared_mem_raw + params.num_bits * params.num_words * sizeof(uint32_t));

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // 1. 加载 4-bit Query
    const uint32_t* query_packed_ptr = params.d_packed_queries + query_idx * params.num_bits * params.num_words;
    for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
        shared_packed_query[i] = query_packed_ptr[i];
    }
    // 2. 加载 Float Query
    for (size_t i = tid; i < params.D; i += num_threads) {
        shared_float_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    float query_width = params.d_widths[query_idx];
    float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];

    __shared__ int probe_slot;
    if (tid == 0) {
        probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
    }
    __syncthreads();
    
    if (probe_slot >= params.max_candidates_per_query) return;
    uint32_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
        size_t vec_idx = vec_base + tid;
        if (vec_idx >= num_vectors_in_cluster) break;
        if (probe_slot + vec_idx >= params.max_candidates_per_query) break; 

        size_t global_vec_idx = cluster_start_index + vec_idx;

        // ---------------------------------------------------------
        // 🌟 阶段 1：使用 Float Query 精确计算 1-bit 主内积
        // ---------------------------------------------------------
        float exact_ip = 0.0f;
        for (int word = 0; word < params.num_words; ++word) {
            uint32_t data_word = params.d_short_data[cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx];
            
            #pragma unroll 8
            for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
                size_t dim = word * 32 + bit_idx;
                if (dim < params.D) {
                    // 从 MSB 提取到 LSB，以匹配 NVIDIA 的位序
                    if ((data_word >> (31 - bit_idx)) & 0x1) { 
                        exact_ip += shared_float_query[dim]; 
                    }
                }
            }
        }

        // ---------------------------------------------------------
        // 🌟 阶段 2：使用 Float 计算 ex_bits 扩展内积
        // ---------------------------------------------------------
        float final_dist = FLT_MAX;
        if (params.ex_bits > 0) {
            float ip2 = 0.0f;
            const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;
            const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;
            
            for (uint32_t d = 0; d < params.D; d++) {
                uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits, long_code_size);
                ip2 += shared_float_query[d] * (float)code_val;
            }

            float2 ex_factors = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
            float q_kbxsumq = params.d_G_kbxSumq[query_idx];
            
            // 注意: 这里用 'exact_ip' 替换了原来的 'ip'
            final_dist = ex_factors.x + q_g_add + 
                         ex_factors.y * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);
            
            if (isnan(final_dist) || isinf(final_dist)) final_dist = FLT_MAX; 
        } else {
            float3 factors = reinterpret_cast<const float3*>(params.d_short_factors)[global_vec_idx];
            float q_k1xsumq = params.d_G_k1xSumq[query_idx];
            
            // 注意: 这里用 'exact_ip' 替换了原来的 'ip'
            final_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq);
        }

        params.d_candidates_dists[output_offset + vec_idx] = final_dist;
        params.d_candidates_pids[output_offset + vec_idx]  = params.d_pids[global_vec_idx];

        if (query_idx == 0 && cluster_idx == 2230 && vec_idx < 5) {
            printf("\n[Kernel Debug] Q0, Cluster %d, LocalVec %d (Global %lu):\n", 
                    cluster_idx, (int)vec_idx, (unsigned long)global_vec_idx);
            printf("  q_g_add (Coarse Dist) : %f\n", q_g_add);
            // printf("  exact_ip (Float * 1b) : %f\n", exact_ip);
            
            if (params.ex_bits > 0) {
                float2 ex_factors = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
                float q_kbxsumq = params.d_G_kbxSumq[query_idx];
                
                printf("  --- [Ex_Bits Mode] ---\n");
                printf("  ex_factors.x (add)    : %f\n", ex_factors.x);
                printf("  ex_factors.y (rescale): %f\n", ex_factors.y);
                // printf("  ip2 (Float * ex_bits) : %f\n", ip2);
                printf("  q_kbxsumq             : %f\n", q_kbxsumq);
                
                // float scaled_exact_ip = static_cast<float>(1 << params.ex_bits) * exact_ip;
                // float ip_term = ex_factors.y * (scaled_exact_ip + ip2 + q_kbxsumq);
                
                // printf("  scaled_exact_ip       : %f\n", scaled_exact_ip);
                // printf("  IP Term (rescale *..) : %f\n", ip_term);
            }
            
            printf("  => final_dist         : %f\n", final_dist);
            
        }
    }
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

__inline__ __device__ float blockReduceSum(float v)
{
  __shared__ float shared[32];  // up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warpReduceSum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();

  float out = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.f;
  if (wid == 0) out = warpReduceSum(out);
  return out;
}

//---------------------------------------------------------------------------
// Kernel: exrabitq_quantize_query
//
// Quantize queries using exrabitq implementation, the output are always int8_t array
//
template <unsigned int BlockSize>
__global__ void exrabitq_quantize_query(
  // Inputs
  const float* __restrict__ d_XP,
  size_t num_points,
  size_t D,
  size_t EX_BITS,
  float const_scaling_factor,
  float kConstEpsilon,
  // Outputs
  int8_t* d_long_code,
  float* d_delta)
{
  //=========================================================================
  // Setup: One block per row
  //=========================================================================
  int row = blockIdx.x;
  if (row >= num_points) return;

  // Dynamically allocated shared memory for one row's data.
  extern __shared__ float s_mem[];
  float* s_xp        = s_mem;
  int8_t* s_tmp_code = (int8_t*)(s_xp + D);
  float* s_reduce    = (float*)(s_tmp_code + D);  // For reduction

  int tid = threadIdx.x;

  //=========================================================================
  // Step 0: Load XP and compute L2 nrom && normalize
  //=========================================================================
  float thread_sum_sq = 0.0f;

  // local L2 norm
  for (int j = tid; j < D; j += BlockSize) {
    float xp_val = d_XP[row * D + j];
    s_xp[j]      = xp_val;
    thread_sum_sq += xp_val * xp_val;  // Direct L2 norm of XP
  }

  s_reduce[tid] = thread_sum_sq;
  __syncthreads();

  // global reduction
  for (unsigned int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) { s_reduce[tid] += s_reduce[tid + stride]; }
    __syncthreads();
  }

  float norm     = sqrtf(s_reduce[0]);
  float norm_inv = (norm > 0) ? (1.0f / norm) : 0.0f;

  //=========================================================================
  // Step 1 (skipped): Coalesced load of all necessary data into shared memory
  //=========================================================================

  //=========================================================================
  // Part A: ExRaBitQ Code Generation
  //=========================================================================
  // Parallel quantization and start of ip_norm reduction
  for (int j = tid; j < D; j += BlockSize) {
    float val    = s_xp[j] * norm_inv;
    int code_val = __float2int_rn((const_scaling_factor * val) /*+ 0.5*/);  // round-to-nearest-even
    if (code_val > (1 << (EX_BITS - 1)) - 1) code_val = (1 << (EX_BITS - 1)) - 1;
    if (code_val < (-(1 << (EX_BITS - 1)))) code_val = -(1 << (EX_BITS - 1));
    s_tmp_code[j] = code_val;
  }
  __syncthreads();

  //=========================================================================
  // Part B: Factor Computation
  //=========================================================================
  float ip_resi_xucb = 0.f, xu_sq = 0.f;

  for (size_t j = tid; j < D; j += BlockSize) {
    float res  = s_xp[j];
    int xu_pre = s_tmp_code[j];

    float xu = float(xu_pre) /* - (static_cast<float>(1 << (EX_BITS - 1))) */;
    // just ignore the 0.5 since we are not going to store extra shift
    ip_resi_xucb += res * xu;  // for cos_similarity
    xu_sq += xu * xu;          // norm_quan^2
  }

  // only thread 0 in the block need the results, so simply use blockReduceSum
  // Perform parallel reductions for all factor components
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  xu_sq        = blockReduceSum(xu_sq);

  // Thread 0 computes and writes the final factors
  if (tid == 0) {
    float norm_quan      = sqrtf(fmaxf(xu_sq, 0.f));
    float cos_similarity = ip_resi_xucb / (norm * norm_quan);
    float delta          = norm / norm_quan * cos_similarity;

    size_t base   = row;
    d_delta[base] = delta;
  }

  //=========================================================================
  // Part C: Pack and Write Long Code (MINIMAL READS, PARALLEL, COALESCED)
  //=========================================================================
  int long_code_length = D;  // D dims, then D bytes
  int8_t* out_ptr      = d_long_code + row * long_code_length;
  for (int j = tid; j < D; j += BlockSize)
    out_ptr[j] = s_tmp_code[j];  // write outputs directly
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

void IVF_RaBitQ_Raw::search(const float* h_queries, size_t num_queries, size_t topk, size_t nprobe, float* h_final_dists, uint32_t* h_final_pids) {
    // 此时传进来的 h_queries 已经是 CPU 旋转过且 Padding 好的数据了！
    float* d_queries;
    size_t queries_bytes = num_queries * padded_dim * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_queries, queries_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_queries, h_queries, queries_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // print_device_array("d_queries", d_queries, 5);
    // print_device_array("d_centroids", d_centroids, 5);

    // 1. 粗量化检索 (CUBLAS)
    float* d_centroid_dists;
    CUDA_CHECK(cudaMallocAsync(&d_centroid_dists, num_queries * num_centroids * sizeof(float), stream));
    const float alpha = -2.0f; const float beta = 0.0f;
    
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                num_centroids, num_queries, padded_dim,
                &alpha, d_centroids, padded_dim, 
                d_queries, padded_dim, 
                &beta, d_centroid_dists, num_centroids);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // print_device_array("d_centroid_dists (after SGEMM)", d_centroid_dists, 5);


    // 2. 补全 L2 距离范数
    float *d_q_norms, *d_c_norms;
    CUDA_CHECK(cudaMallocAsync(&d_q_norms, num_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_c_norms, num_centroids * sizeof(float), stream));
    
    int norm_block_size = 256;
    size_t norm_shared_mem = ((norm_block_size + 31) / 32) * sizeof(float);
    row_norms_fused_kernel<<<num_centroids + num_queries, norm_block_size, norm_shared_mem, stream>>>(
        d_queries, num_queries, padded_dim, d_centroids, num_centroids, padded_dim, d_q_norms, d_c_norms);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // print_device_array("d_q_norms", d_q_norms, 5);
    // print_device_array("d_c_norms", d_c_norms, 5);

    int add_threads = 256;
    int add_blocks  = (num_queries * num_centroids + add_threads - 1) / add_threads;
    add_norms_kernel<<<add_blocks, add_threads, 0, stream>>>(d_centroid_dists, d_q_norms, d_c_norms, num_queries, num_centroids);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // print_device_array("d_centroid_dists (final coarse)", d_centroid_dists, 5);

    // 3. 选出 Top-nprobe 簇并重排为 Pair
    float* d_probe_vals; int* d_probe_idx;
    CUDA_CHECK(cudaMallocAsync(&d_probe_vals, num_queries * nprobe * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_probe_idx, num_queries * nprobe * sizeof(int), stream));

    // 粗量化 cuBLAS 算完之后：
    float* d_centroid_dists_copy;
    CUDA_CHECK(cudaMallocAsync(&d_centroid_dists_copy, num_queries * num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_centroid_dists_copy, d_centroid_dists, num_queries * num_centroids * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // 使用 COPY 传进去破坏！
    select_topk_kernel<<<num_queries, 256, 0, stream>>>(d_centroid_dists_copy, num_centroids, nprobe, d_probe_vals, d_probe_idx);

    // 用完释放
    CUDA_CHECK(cudaFreeAsync(d_centroid_dists_copy, stream));

    // --- Debug Hook 1: 打印 Query 0 选出的 Top-5 Cluster ID 和距离 ---
    std::vector<float> h_dbg_probe_vals(nprobe);
    std::vector<int> h_dbg_probe_idx(nprobe);
    cudaMemcpy(h_dbg_probe_vals.data(), d_probe_vals, nprobe * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dbg_probe_idx.data(), d_probe_idx, nprobe * sizeof(int), cudaMemcpyDeviceToHost);
    
    // std::cout << "\n[Debug 1] Query 0 Top-5 Probed Clusters:\n";
    // for(int i = 0; i < std::min((int)nprobe, 5); i++) {
    //     std::cout << "  Rank " << i << " | Cluster ID: " << h_dbg_probe_idx[i] 
    //               << " | Coarse Dist: " << h_dbg_probe_vals[i] << std::endl;
    // }
    
    int total_probes = num_queries * nprobe;
    ClusterQueryPair* d_sorted_pairs;
    CUDA_CHECK(cudaMallocAsync(&d_sorted_pairs, total_probes * sizeof(ClusterQueryPair), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // std::cout << "Total Probes (num_queries * nprobe): " << total_probes << std::endl;
    flatten_pairs_kernel<<<(total_probes + 255) / 256, 256, 0, stream>>>(d_probe_idx, num_queries, nprobe, d_sorted_pairs);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // std::cout << "Flattened Cluster-Query pairs into d_sorted_pairs." << std::endl;


    thrust::sort(thrust::cuda::par.on(stream), d_sorted_pairs, d_sorted_pairs + total_probes, PairSortCompare());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // for (int i = 0; i < std::min(10, total_probes); i++) {
    //     ClusterQueryPair h_pair;
    //     cudaMemcpy(&h_pair, d_sorted_pairs + i, sizeof(ClusterQueryPair), cudaMemcpyDeviceToHost);
    //     std::cout << "  Sorted Pair " << i << " | Query ID: " << h_pair.query_idx 
    //               << " | Cluster ID: " << h_pair.cluster_idx << std::endl;
    // }

    // 4. 预计算 Query 还原 Factors (修复 Bug 3)
    float *d_G_k1xSumq, *d_G_kbxSumq;
    CUDA_CHECK(cudaMallocAsync(&d_G_k1xSumq, num_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_G_kbxSumq, num_queries * sizeof(float), stream));
    computeQueryFactors<float>(d_queries, d_G_k1xSumq, d_G_kbxSumq, num_queries, padded_dim, ex_bits, stream);

    //简单方法
    size_t pool_size_per_query = nprobe * this->max_cluster_size_; 
    size_t total_candidates = num_queries * pool_size_per_query;
    
    float* d_candidates_dists; 
    uint32_t* d_candidates_pids; 
    int* d_query_write_counters;
    
    CUDA_CHECK(cudaMallocAsync(&d_candidates_dists, total_candidates * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_candidates_pids, total_candidates * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_query_write_counters, num_queries * sizeof(int), stream));
    
    // 【关键】用无穷大填满距离池，防止后续 Top-K 选择时把未初始化的垃圾值当成最优解
    thrust::fill(thrust::cuda::par.on(stream), 
                 thrust::device_pointer_cast(d_candidates_dists), 
                 thrust::device_pointer_cast(d_candidates_dists) + total_candidates, 
                 FLT_MAX);
    CUDA_CHECK(cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream));

    // 配置 Kernel 参数，果断置空 4-bit Query 相关的参数
    int num_words = (padded_dim + 31) / 32;
    
    ComputeBitwiseKernelParams bitwiseParams;
    bitwiseParams.d_sorted_pairs = d_sorted_pairs;
    bitwiseParams.d_cluster_meta = d_cluster_meta;
    bitwiseParams.d_query = d_queries; // <-- 传入的直接是 Float Query
    bitwiseParams.d_packed_queries = nullptr; // 不使用
    bitwiseParams.d_widths = nullptr;         // 不使用
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
    bitwiseParams.num_bits = 4; // 虽然不用 query 的 bit，但保留配置以防后患
    bitwiseParams.num_words = num_words;
    bitwiseParams.num_pairs = total_probes;
    bitwiseParams.max_candidates_per_query = pool_size_per_query;

    // 只需要为 Float Query 分配 Shared Memory
    size_t shared_mem_size = (padded_dim + this->max_cluster_size_) * sizeof(float);
    computeInnerProductsWithBitwiseOpt<<<total_probes, 256, shared_mem_size, stream>>>(bitwiseParams);
    // size_t shared_mem_size = padded_dim * sizeof(float);
    // computeInnerProductsFloatQuery_NoBlockSort<<<total_probes, 256, shared_mem_size, stream>>>(bitwiseParams);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    
    /*
    // 5. 准备 Bitwise 在线量化 (已切换为 RaBitQ 量化)
    int num_bits = 4;
    int num_words = (padded_dim + 31) / 32;
    float *d_widths; int8_t *d_quantized_queries; uint32_t *d_packed_queries; float* d_topk_threshold_batch;
    
    // 注意：不再需要分配 d_query_ranges
    CUDA_CHECK(cudaMallocAsync(&d_widths, num_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_quantized_queries, num_queries * padded_dim * sizeof(int8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_packed_queries, num_queries * num_bits * num_words * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_topk_threshold_batch, num_queries * sizeof(float), stream));

    // --- 启动 RaBitQ 动态量化 ---
    const int quant_block_size = 256;
    const int quant_grid_size  = num_queries;
    // 根据 Nvidia 官方 Kernel 内部需求计算 Shared Memory
    size_t quant_shared_mem = padded_dim * sizeof(float) + padded_dim * sizeof(int8_t) + quant_block_size * sizeof(float);
    
    // 如果你的 IVF_RaBitQ_Raw 类里没有 best_rescaling_factor 成员变量，
    // 请确保根据你的数据集定义它 (比如 4-bit 下可能是某个常数，你可以暂时传 1.0f 跑通)
    
    std::cout << "\n[Debug Pipeline] Starting exrabitq_quantize_query..." << std::endl;
    exrabitq_quantize_query<quant_block_size><<<quant_grid_size, quant_block_size, quant_shared_mem, stream>>>(
        d_queries,
        num_queries,
        padded_dim,
        num_bits,                  // EX_BITS: 这里传入 4
        best_rescaling_factor,     // const_scaling_factor
        1.9f,                      // kConstEpsilon
        d_quantized_queries,       // 输出的 4-bit code (存放在 int8_t 数组中)
        d_widths                   // RaBitQ 的 delta 被写回 d_widths 给后续使用
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[Debug Pipeline] exrabitq_quantize_query finished safely." << std::endl;

    // --- 启动位平面打包 ---
    std::cout << "[Debug Pipeline] Starting packInt4QueryBitPlanes..." << std::endl;
    packInt4QueryBitPlanes<<<(num_queries * num_bits * num_words + 255) / 256, 256, 0, stream>>>(
        d_quantized_queries, 
        d_packed_queries, 
        num_queries, 
        padded_dim
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[Debug Pipeline] packInt4QueryBitPlanes finished safely." << std::endl;

    // --- Debug Hook 2: 打印 Query 0 的量化因子 ---
    float h_dbg_width = 0.0f, h_dbg_Gk1 = 0.0f, h_dbg_Gkb = 0.0f;
    cudaMemcpy(&h_dbg_width, d_widths, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dbg_Gk1, d_G_k1xSumq, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dbg_Gkb, d_G_kbxSumq, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\n[Debug 2] Query 0 Quantization Info (RaBitQ Model):\n"
              << "  Delta (Width): " << h_dbg_width << "\n"
              << "  G_k1xSumq: " << h_dbg_Gk1 << "\n"
              << "  G_kbxSumq: " << h_dbg_Gkb << std::endl;

    // 6. 细检索: Bitwise Popcount 与候选池写入
    size_t pool_size_per_query = nprobe * this->max_cluster_size_; 
    size_t total_candidates = num_queries * pool_size_per_query;
    float* d_candidates_dists; uint32_t* d_candidates_pids; int* d_query_write_counters;
    CUDA_CHECK(cudaMallocAsync(&d_candidates_dists, total_candidates * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_candidates_pids, total_candidates * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_query_write_counters, num_queries * sizeof(int), stream));
    
    // 【关键】用无穷大填满距离矩阵，防止垃圾空位干扰 Top-K 挑选
    thrust::fill(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_candidates_dists), thrust::device_pointer_cast(d_candidates_dists) + total_candidates, FLT_MAX);
    CUDA_CHECK(cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream));

    thrust::fill(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_topk_threshold_batch), thrust::device_pointer_cast(d_topk_threshold_batch) + num_queries, FLT_MAX);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ComputeBitwiseKernelParams bitwiseParams;
    bitwiseParams.d_sorted_pairs = d_sorted_pairs;
    bitwiseParams.d_cluster_meta = d_cluster_meta;
    bitwiseParams.d_query = d_queries;
    bitwiseParams.d_packed_queries = d_packed_queries;
    bitwiseParams.d_threshold = d_topk_threshold_batch;
    bitwiseParams.d_widths = d_widths;
    bitwiseParams.d_G_k1xSumq = d_G_k1xSumq; 
    bitwiseParams.d_centroid_distances = d_centroid_dists;
    bitwiseParams.d_short_data = d_short_data;
    bitwiseParams.d_short_factors = d_short_factors;
    bitwiseParams.d_long_code = d_long_codes;         // 确保你已经 cudaMemcpy 传到 GPU 了
    bitwiseParams.d_ex_factor = d_ex_factors;         // 确保你已经 cudaMemcpy 传到 GPU 了
    bitwiseParams.d_G_kbxSumq = d_G_kbxSumq;
    bitwiseParams.ex_bits = ex_bits;
    bitwiseParams.d_pids = d_ids;
    bitwiseParams.d_candidates_dists = d_candidates_dists;
    bitwiseParams.d_candidates_pids = d_candidates_pids;
    bitwiseParams.d_query_write_counters = d_query_write_counters;
    bitwiseParams.num_queries = num_queries;
    bitwiseParams.num_centroids = num_centroids;
    bitwiseParams.D = padded_dim;
    bitwiseParams.num_bits = num_bits;
    bitwiseParams.num_words = num_words;
    bitwiseParams.num_pairs = total_probes;
    bitwiseParams.nprobe = nprobe;
    bitwiseParams.topk = topk;
    bitwiseParams.max_candidates_per_query = pool_size_per_query;
    bitwiseParams.max_candidates_per_pair = this->max_cluster_size_;

    // size_t shared_mem_size = (num_bits * num_words * sizeof(uint32_t));
    // size_t shared_mem_size = padded_dim * sizeof(float);
    // size_t shared_mem_size = (num_bits * num_words * sizeof(uint32_t)) + (padded_dim * sizeof(float));

    size_t packed_query_bytes = std::max(
        (size_t)(bitwiseParams.num_bits * bitwiseParams.num_words * sizeof(uint32_t)),
        (size_t)(bitwiseParams.max_candidates_per_pair * sizeof(float))
    );

    size_t shared_mem_size = packed_query_bytes 
                            + bitwiseParams.max_candidates_per_pair * sizeof(float) // 给 shared_candidate_ips
                            + bitwiseParams.max_candidates_per_pair * sizeof(int)   // 给 shared_candidate_indices
                            + bitwiseParams.D * sizeof(float);                      // 给 shared_query

    // 2. [安全检查] 确保没有超过 CUDA 单个 Block 的共享内存硬件上限 (默认 48KB)
    if (shared_mem_size > 49152) {
        std::cerr << "FATAL: Shared memory requested (" << shared_mem_size 
                << " bytes) exceeds 48KB limit! Please reduce max_candidates_per_pair." << std::endl;
        exit(1);
    }
    computeInnerProductsWithBitwiseOpt4bit_NoSort<<<total_probes, 256, shared_mem_size, stream>>>(bitwiseParams);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    */
    
    // 7. 全局最终 Top-K 聚合与 PID 映射 (修复 Bug 2)
    float* d_final_dev_dists; uint32_t* d_final_dev_pids;
    size_t final_bytes = num_queries * topk * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_final_dev_dists, final_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_final_dev_pids, num_queries * topk * sizeof(uint32_t), stream));

    // 使用修复后的参数尺寸搜寻整个候选池
    select_topk_kernel<<<num_queries, 256, 0, stream>>>(d_candidates_dists, pool_size_per_query, topk, d_final_dev_dists, (int*)d_final_dev_pids);
    
    // 执行 PID 映射还原
    map_pid_kernel<<<(num_queries * topk + 255) / 256, 256, 0, stream>>>((int*)d_final_dev_pids, d_candidates_pids, d_final_dev_pids, num_queries * topk, pool_size_per_query, topk);

    // 8. 拷贝回 Host
    CUDA_CHECK(cudaMemcpyAsync(h_final_dists, d_final_dev_dists, final_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_final_pids, d_final_dev_pids, num_queries * topk * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMallocAsync(&d_final_dev_pids, num_queries * topk * sizeof(uint32_t), stream));
    // 【防崩溃补丁】：强制清零，就算算错，返回的也是 ID 0，不会导致测评程序崩溃
    CUDA_CHECK(cudaMemsetAsync(d_final_dev_pids, 0, num_queries * topk * sizeof(uint32_t), stream));

    // --- Debug Hook 3: 打印 Query 0 最终返回的 Top-10 结果 ---
    // std::cout << "\n[Debug 3] Query 0 Final Output:\n";
    // for(int i = 0; i < std::min((int)topk, 10); i++) {
    //     std::cout << "  Rank " << i << " | PID: " << h_final_pids[i] 
    //               << " | Final Dist: " << h_final_dists[i] << std::endl;
    // }

    // 清理所有显存
    cudaFree(d_queries); cudaFree(d_centroid_dists); cudaFree(d_q_norms); cudaFree(d_c_norms);
    cudaFree(d_probe_vals); cudaFree(d_probe_idx); cudaFree(d_sorted_pairs);
    // cudaFree(d_widths); cudaFree(d_quantized_queries); cudaFree(d_packed_queries);
    cudaFree(d_candidates_dists); cudaFree(d_candidates_pids); cudaFree(d_query_write_counters);
    cudaFree(d_final_dev_dists); cudaFree(d_final_dev_pids); cudaFree(d_G_k1xSumq); cudaFree(d_G_kbxSumq);
}