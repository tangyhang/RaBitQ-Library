#pragma once

#include <omp.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {
namespace scalar_impl {
template <typename T>
void scalar_quantize_normal(
    T* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    float one_over_delta = 1.0F / delta;

    ConstRowMajorArrayMap<float> v0(vec0, 1, static_cast<long>(dim));
    RowMajorArrayMap<T> res(result, 1, dim);

    // round to nearest integer, then cast to integer
    res = ((v0 - lo) * one_over_delta).round().template cast<T>();
}

template <typename T>
void scalar_quantize_optimized(
    T* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    scalar_quantize_normal(result, vec0, dim, lo, delta);
}

template <>
inline void scalar_quantize_optimized<uint8_t>(
    uint8_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    float one_over_delta = 1 / delta;
    for (size_t i = 0; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

template <>
inline void scalar_quantize_optimized<uint16_t>(
    uint16_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    float one_over_delta = 1 / delta;
    for (size_t i = 0; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}
}  // namespace scalar_impl

template <typename T>
inline void vec_rescale(T* data, size_t dim, T val) {
    RowMajorArrayMap<T> data_arr(data, 1, dim);
    data_arr *= val;
}

template <typename T>
inline T euclidean_sqr(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return (v0 - v1).dot(v0 - v1);
}

template <typename T>
inline T dot_product_dis(
    const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim
) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return 1 - v0.dot(v1);
}

template <typename T>
inline T l2norm_sqr(const T* __restrict__ vec0, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    return v0.dot(v0);
}

template <typename T>
inline T dot_product(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return v0.dot(v1);
}

template <typename T>
inline T normalize_vec(
    const T* __restrict__ vec, const T* __restrict__ centroid, T* res, T dist2c, size_t dim
) {
    RowMajorArrayMap<T> r(res, 1, dim);
    if (dist2c > 1e-5) {
        ConstRowMajorArrayMap<T> v(vec, 1, dim);
        ConstRowMajorArrayMap<T> c(centroid, 1, dim);
        r = (v - c) * (1 / dist2c);
        return r.sum();
    }
    T value = 1.0 / std::sqrt(static_cast<T>(dim));
    r = value;
    return static_cast<T>(dim) * value;
}

// pack 0/1 data to usigned integer
template <typename T>
inline void pack_binary(
    const int* __restrict__ binary_code, T* __restrict__ compact_code, size_t length
) {
    constexpr size_t kTypeBits = sizeof(T) * 8;

    for (size_t i = 0; i < length; i += kTypeBits) {
        T cur = 0;
        for (size_t j = 0; j < kTypeBits; ++j) {
            cur |= (static_cast<T>(binary_code[i + j]) << (kTypeBits - 1 - j));
        }
        *compact_code = cur;
        ++compact_code;
    }
}

template <typename T>
inline void data_range(const T* __restrict__ vec0, size_t dim, T& lo, T& hi) {
    ConstRowMajorArrayMap<T> v0(vec0, 1, dim);
    lo = v0.minCoeff();
    hi = v0.maxCoeff();
}

template <typename T, typename TD>
void scalar_quantize(
    T* __restrict__ result, const TD* __restrict__ vec0, size_t dim, TD lo, TD delta
) {
    assert_integral<T>();
    scalar_impl::scalar_quantize_optimized(result, vec0, dim, lo, delta);
}

template <typename T>
inline std::vector<T> compute_centroid(
    const T* data, size_t num_points, size_t dim, size_t num_threads
) {
    omp_set_num_threads(static_cast<int>(num_threads));
    std::vector<std::vector<T>> all_results(num_threads, std::vector<T>(dim, 0));

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        std::vector<T>& cur_results = all_results[tid];
        const T* cur_data = data + (dim * i);
        for (size_t k = 0; k < dim; ++k) {
            cur_results[k] += cur_data[k];
        }
    }

    std::vector<T> centroid(dim, 0);
    for (auto& one_res : all_results) {
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] += one_res[i];
        }
    }
    T inv_num_points = 1 / static_cast<T>(num_points);

    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = centroid[i] * inv_num_points;
    }

    return centroid;
}

template <typename T>
inline PID exact_nn(
    const T* data,
    const T* query,
    size_t num_points,
    size_t dim,
    size_t num_threads,
    T (*dist_func)(const T*, const T*, size_t)
) {
    std::vector<AnnCandidate<T, PID>> best_entries(num_threads);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        AnnCandidate<T, PID>& cur_entry = best_entries[tid];
        const T* cur_data = data + (dim * i);

        T distance = dist_func(cur_data, query, dim);
        if (distance < cur_entry.distance) {
            cur_entry.id = static_cast<PID>(i);
            cur_entry.distance = distance;
        }
    }

    PID nearest_neighbor = 0;
    T min_dist = std::numeric_limits<T>::max();
    for (auto& candi : best_entries) {
        if (candi.distance < min_dist) {
            nearest_neighbor = candi.id;
            min_dist = candi.distance;
        }
    }
    return nearest_neighbor;
}

namespace excode_ipimpl {

// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
inline float ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0;

    for (size_t i = 0; i < dim; i += 16) {
        uint16_t mask = *reinterpret_cast<const uint16_t*>(compact_code);
        
        for (size_t j = 0; j < 16; ++j) {
            if (mask & (1 << (15 - j))) {
                result += query[j];
            }
        }

        compact_code += 2;
        query += 16;
    }
    return result;
}

inline float ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0;
    const uint8_t mask = 0b00000011;

    for (size_t i = 0; i < dim; i += 64) {
        for (size_t j = 0; j < 16; ++j) {
            uint8_t byte = compact_code[j];
            
            // Extract 2-bit values from each byte
            for (size_t k = 0; k < 4; ++k) {
                uint8_t val = (byte >> (k * 2)) & mask;
                size_t idx = j * 4 + k;
                
                if (idx < 16) result += query[i + idx] * val;
                if (idx < 16) result += query[i + 16 + idx] * ((byte >> (k * 2)) & mask);
                if (idx < 16) result += query[i + 32 + idx] * ((byte >> (k * 2)) & mask);
                if (idx < 16) result += query[i + 48 + idx] * ((byte >> (k * 2)) & mask);
            }
        }
        compact_code += 16;
    }

    return result;
}

inline float ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0;
    const uint8_t mask = 0b11;

    for (size_t i = 0; i < dim; i += 64) {
        uint8_t compact2[16];
        std::memcpy(compact2, compact_code, 16);
        compact_code += 16;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        for (size_t j = 0; j < 16; ++j) {
            uint8_t byte = compact2[j];
            
            for (size_t k = 0; k < 4; ++k) {
                uint8_t val = (byte >> (k * 2)) & mask;
                size_t idx = j * 4 + k;
                
                // Add top bit if needed
                if (top_bit & (1LL << idx)) {
                    val |= 0b100;
                }
                
                if (idx < 16) result += query[i + idx] * val;
                if (idx < 16) result += query[i + 16 + idx] * val;
                if (idx < 16) result += query[i + 32 + idx] * val;
                if (idx < 16) result += query[i + 48 + idx] * val;
            }
        }
    }

    return result;
}

inline float ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0.0F;
    constexpr int64_t kMask = 0x0f0f0f0f0f0f0f0f;
    
    for (size_t i = 0; i < dim; i += 16) {
        int64_t compact = *reinterpret_cast<const int64_t*>(compact_code);
        int64_t code0 = compact & kMask;
        int64_t code1 = (compact >> 4) & kMask;

        for (size_t j = 0; j < 8; ++j) {
            int8_t val0 = static_cast<int8_t>((code0 >> (j * 8)) & 0xff);
            int8_t val1 = static_cast<int8_t>((code1 >> (j * 8)) & 0xff);
            
            result += query[i + j] * val0;
            result += query[i + j + 8] * val1;
        }
        
        compact_code += 8;
    }
    return result;
}

inline float ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0.0F;
    const uint8_t mask = 0b1111;

    for (size_t i = 0; i < dim; i += 64) {
        uint8_t compact4_1[16];
        uint8_t compact4_2[16];
        std::memcpy(compact4_1, compact_code, 16);
        std::memcpy(compact4_2, compact_code + 16, 16);
        compact_code += 32;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        for (size_t j = 0; j < 16; ++j) {
            uint8_t byte1 = compact4_1[j];
            uint8_t byte2 = compact4_2[j];
            
            for (size_t k = 0; k < 2; ++k) {
                uint8_t val1 = (byte1 >> (k * 4)) & mask;
                uint8_t val2 = (byte2 >> (k * 4)) & mask;
                
                size_t idx = j * 2 + k;
                
                // Add top bit if needed
                if (top_bit & (1LL << idx)) {
                    val1 |= 0b10000;
                    val2 |= 0b10000;
                }
                
                if (idx < 16) result += query[i + idx] * val1;
                if (idx < 16) result += query[i + 16 + idx] * val1;
                if (idx < 16) result += query[i + 32 + idx] * val2;
                if (idx < 16) result += query[i + 48 + idx] * val2;
            }
        }
    }
    return result;
}

inline float ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0.0F;
    const uint8_t mask6 = 0b00111111;

    for (size_t i = 0; i < dim; i += 64) {
        uint8_t cpt1[16];
        uint8_t cpt2[16];
        uint8_t cpt3[16];
        std::memcpy(cpt1, compact_code, 16);
        std::memcpy(cpt2, compact_code + 16, 16);
        std::memcpy(cpt3, compact_code + 32, 16);
        compact_code += 48;

        for (size_t j = 0; j < 16; ++j) {
            uint8_t val1 = cpt1[j] & mask6;
            uint8_t val2 = cpt2[j] & mask6;
            uint8_t val3 = cpt3[j] & mask6;
            
            // Extract 2-bit values from the upper bits
            uint8_t val4 = ((cpt1[j] >> 6) & 0b11) | 
                          ((cpt2[j] >> 4) & 0b1100) | 
                          ((cpt3[j] >> 2) & 0b110000);
            
            if (j < 16) result += query[i + j] * val1;
            if (j < 16) result += query[i + 16 + j] * val2;
            if (j < 16) result += query[i + 32 + j] * val3;
            if (j < 16) result += query[i + 48 + j] * val4;
        }
    }
    return result;
}

inline float ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0.0F;
    const uint8_t mask6 = 0b00111111;

    for (size_t i = 0; i < dim; i += 64) {
        uint8_t cpt1[16];
        uint8_t cpt2[16];
        uint8_t cpt3[16];
        std::memcpy(cpt1, compact_code, 16);
        std::memcpy(cpt2, compact_code + 16, 16);
        std::memcpy(cpt3, compact_code + 32, 16);
        compact_code += 48;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        for (size_t j = 0; j < 16; ++j) {
            uint8_t val1 = cpt1[j] & mask6;
            uint8_t val2 = cpt2[j] & mask6;
            uint8_t val3 = cpt3[j] & mask6;
            
            // Extract 2-bit values from the upper bits
            uint8_t val4 = ((cpt1[j] >> 6) & 0b11) | 
                          ((cpt2[j] >> 4) & 0b1100) | 
                          ((cpt3[j] >> 2) & 0b110000);
            
            // Add top bit if needed
            if (top_bit & (1LL << j)) {
                val1 |= 0b1000000;
                val2 |= 0b1000000;
                val3 |= 0b1000000;
                val4 |= 0b1000000;
            }
            
            if (j < 16) result += query[i + j] * val1;
            if (j < 16) result += query[i + 16 + j] * val2;
            if (j < 16) result += query[i + 32 + j] * val3;
            if (j < 16) result += query[i + 48 + j] * val4;
        }
    }
    return result;
}

// inner product between float type and int type vectors
template <typename TF, typename TI>
inline TF ip_fxi(const TF* __restrict__ vec0, const TI* __restrict__ vec1, size_t dim) {
    static_assert(std::is_floating_point_v<TF>, "TF must be an floating type");
    static_assert(std::is_integral_v<TI>, "TI must be an integeral type");

    ConstVectorMap<TF> v0(vec0, dim);
    ConstVectorMap<TI> v1(vec1, dim);
    return v0.dot(v1.template cast<TF>());
}
}  // namespace excode_ipimpl

using ex_ipfunc = float (*)(const float*, const uint8_t*, size_t);

inline ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 1) {
        // when ex_bits = 0, we do not use it
        return excode_ipimpl::ip16_fxu1_avx;
    }
    if (ex_bits == 2) {
        return excode_ipimpl::ip64_fxu2_avx;
    }
    if (ex_bits == 3) {
        return excode_ipimpl::ip64_fxu3_avx;
    }
    if (ex_bits == 4) {
        return excode_ipimpl::ip16_fxu4_avx;
    }
    if (ex_bits == 5) {
        return excode_ipimpl::ip64_fxu5_avx;
    }
    if (ex_bits == 6) {
        return excode_ipimpl::ip64_fxu6_avx;
    }
    if (ex_bits == 7) {
        return excode_ipimpl::ip64_fxu7_avx;
    }
    if (ex_bits == 8) {
        return excode_ipimpl::ip_fxi;
    }

    std::cerr << "Bad IP function for IVF\n";
    exit(1);
}

static inline uint32_t reverse_bits(uint32_t n) {
    n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
    n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
    n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
    n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
    return n;
}

static inline uint64_t reverse_bits_u64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555) | ((n << 1) & 0xaaaaaaaaaaaaaaaa);
    n = ((n >> 2) & 0x3333333333333333) | ((n << 2) & 0xcccccccccccccccc);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0f) | ((n << 4) & 0xf0f0f0f0f0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff00ff00ff) | ((n << 8) & 0xff00ff00ff00ff00);
    n = ((n >> 16) & 0x0000ffff0000ffff) | ((n << 16) & 0xffff0000ffff0000);
    n = ((n >> 32) & 0x00000000ffffffff) | ((n << 32) & 0xffffffff00000000);
    return n;
}

static inline void new_transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    for (size_t i = 0; i < padded_dim; i += 64) {
        for (size_t j = 0; j < b_query; ++j) {
            uint64_t v = 0;
            for (size_t k = 0; k < 64; ++k) {
                if (q[k] & (1 << (15 - j))) {
                    v |= (1ULL << k);
                }
            }
            tq[b_query - j - 1] = v;
        }
        tq += b_query;
        q += 64;
    }
}

inline float mask_ip_x0_q_old(const float* query, const uint64_t* data, size_t padded_dim) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

    float sum = 0.0f;
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        
        for (size_t j = 0; j < 64; ++j) {
            if (bits & (1ULL << j)) {
                sum += it_query[j];
            }
        }

        it_data++;
        it_query += 64;
    }
    return sum;
}

inline float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;

    float sum = 0.0f;
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        
        for (size_t j = 0; j < 64; ++j) {
            if (bits & (1ULL << j)) {
                sum += it_query[j];
            }
        }

        ++it_data;
        it_query += 64;
    }

    return sum;
}

inline float ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

    size_t ip = 0;
    size_t ppc = 0;

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t x = *static_cast<const uint64_t*>(it_data);
        ppc += __builtin_popcountll(x);

        for (size_t j = 0; j < b_query; ++j) {
            uint64_t y = *static_cast<const uint64_t*>(it_query);
            ip += (__builtin_popcountll(x & y) << j);
            it_query++;
        }
        it_data++;
    }

    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}

static inline uint32_t ip_bin_bin(const uint64_t* q, const uint64_t* d, size_t padded_dim) {
    uint64_t ret = 0;
    size_t iter = padded_dim / 64;
    for (size_t i = 0; i < iter; ++i) {
        ret += __builtin_popcountll((*d) & (*q));
        q++;
        d++;
    }
    return ret;
}

inline uint32_t ip_byte_bin(
    const uint64_t* q, const uint64_t* d, size_t padded_dim, size_t b_query
) {
    uint32_t ret = 0;
    size_t offset = (padded_dim / 64);
    for (size_t i = 0; i < b_query; i++) {
        ret += (ip_bin_bin(q, d, padded_dim) << i);
        q += offset;
    }
    return ret;
}

inline size_t popcount(const uint64_t* __restrict__ d, size_t length) {
    size_t ret = 0;
    for (size_t i = 0; i < length / 64; ++i) {
        ret += __builtin_popcountll((*d));
        ++d;
    }
    return ret;
}

template <typename T>
RowMajorMatrix<T> random_gaussian_matrix(size_t rows, size_t cols) {
    RowMajorMatrix<T> rand(rows, cols);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, 1);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            rand(i, j) = dist(gen);
        }
    }

    return rand;
}
}  // namespace rabitqlib