#include "cuda_integration.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cmath>
#include <chrono>

#ifndef CURAND_CALL
#define CURAND_CALL(x) do { curandStatus_t status = (x); if (status != CURAND_STATUS_SUCCESS) { exit(EXIT_FAILURE); } } while(0)
#endif

// ============================================================================
// OPTIMAL GPU CONFIGURATION
// ============================================================================
GPUConfig get_optimal_gpu_config() {
    GPUConfig config;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    config.sm_count = prop.multiProcessorCount;
    config.threads_per_block = 256;
    config.blocks_per_sm = 4;
    
    return config;
}

// ============================================================================
// LIGHTWEIGHT PSEUDO-RANDOM NUMBER GENERATOR
// ============================================================================
__device__ __forceinline__ double lcg_random_double(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(state >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ float lcg_random_float(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((state >> 11) * (1.0 / 9007199254740992.0));
}

// ============================================================================
// DEVICE EXPRESSION EVALUATORS
// ============================================================================

__device__ __forceinline__ float eval_expr_device_fp32(
    const TokenType* types,
    const float* constants,
    const int* var_indices,
    const int* op_codes,
    int length,
    const float* vars)
{
    float stack[32];
    int sp = 0;
    
    for (int i = 0; i < length; ++i) {
        TokenType t = types[i];
        
        if (t == TokenType::Number) {
            stack[sp++] = constants[i];
        }
        else if (t == TokenType::Variable) {
            int idx = var_indices[i];
            stack[sp++] = (idx >= 0 && idx < 4) ? vars[idx] : 0.0f;
        }
        else if (t == TokenType::Operator) {
            if (sp < 2) return 0.0f;
            float b = stack[--sp];
            float a = stack[--sp];
            int op = op_codes[i];
            
            float result;
            switch (op) {
                case 0: result = a + b; break;
                case 1: result = a - b; break;
                case 2: result = a * b; break;
                case 3: result = (fabsf(b) > 1e-10f) ? (a / b) : 0.0f; break;
                case 4: result = powf(a, b); break;
                default: result = 0.0f;
            }
            stack[sp++] = result;
        }
        else if (t == TokenType::Function) {
            if (sp < 1) return 0.0f;
            float a = stack[--sp];
            int op = op_codes[i];
            
            float result;
            switch (op) {
                case 10: result = sinf(a); break;
                case 11: result = cosf(a); break;
                case 12: result = (a > 1e-10f) ? log10f(a) : -10.0f; break;
                case 13: result = (a > 1e-10f) ? logf(a) : -23.0f; break;
                case 14: result = expf(fminf(a, 88.0f)); break;
                case 15: result = (a >= 0.0f) ? sqrtf(a) : 0.0f; break;
                case 16: result = tanf(a); break;
                case 17: result = fabsf(a); break;
                default: result = 0.0f;
            }
            stack[sp++] = result;
        }
    }
    
    return (sp > 0) ? stack[0] : 0.0f;
}

__device__ __forceinline__ double eval_expr_device_fp64(
    const TokenType* types,
    const float* constants,
    const int* var_indices,
    const int* op_codes,
    int length,
    const double* vars)
{
    double stack[32];
    int sp = 0;
    
    for (int i = 0; i < length; ++i) {
        TokenType t = types[i];
        
        if (t == TokenType::Number) {
            stack[sp++] = static_cast<double>(constants[i]);
        }
        else if (t == TokenType::Variable) {
            int idx = var_indices[i];
            stack[sp++] = (idx >= 0 && idx < 4) ? vars[idx] : 0.0;
        }
        else if (t == TokenType::Operator) {
            if (sp < 2) return 0.0;
            double b = stack[--sp];
            double a = stack[--sp];
            int op = op_codes[i];
            
            double result;
            switch (op) {
                case 0: result = a + b; break;
                case 1: result = a - b; break;
                case 2: result = a * b; break;
                case 3: result = (fabs(b) > 1e-15) ? (a / b) : 0.0; break;
                case 4: result = pow(a, b); break;
                default: result = 0.0;
            }
            stack[sp++] = result;
        }
        else if (t == TokenType::Function) {
            if (sp < 1) return 0.0;
            double a = stack[--sp];
            int op = op_codes[i];
            
            double result;
            switch (op) {
                case 10: result = sin(a); break;
                case 11: result = cos(a); break;
                case 12: result = (a > 1e-15) ? log10(a) : -15.0; break;
                case 13: result = (a > 1e-15) ? log(a) : -34.5; break;
                case 14: result = exp(fmin(a, 709.0)); break;
                case 15: result = (a >= 0.0) ? sqrt(a) : 0.0; break;
                case 16: result = tan(a); break;
                case 17: result = fabs(a); break;
                default: result = 0.0;
            }
            stack[sp++] = result;
        }
    }
    
    return (sp > 0) ? stack[0] : 0.0;
}

// Native double-constants evaluator (avoids float->double casts)
__device__ __forceinline__ double eval_expr_device_fp64_native(
    const TokenType* types,
    const double* constants,
    const int* var_indices,
    const int* op_codes,
    int length,
    const double* vars)
{
    double stack[32];
    int sp = 0;
    for (int i = 0; i < length; ++i) {
        TokenType t = types[i];
        if (t == TokenType::Number) {
            stack[sp++] = constants[i];
        }
        else if (t == TokenType::Variable) {
            int idx = var_indices[i];
            stack[sp++] = (idx >= 0 && idx < 4) ? vars[idx] : 0.0;
        }
        else if (t == TokenType::Operator) {
            if (sp < 2) return 0.0;
            double b = stack[--sp];
            double a = stack[--sp];
            int op = op_codes[i];
            double result;
            switch (op) {
                case 0: result = a + b; break;
                case 1: result = a - b; break;
                case 2: result = a * b; break;
                case 3: result = (fabs(b) > 1e-15) ? (a / b) : 0.0; break;
                case 4: result = pow(a, b); break;
                default: result = 0.0;
            }
            stack[sp++] = result;
        }
        else if (t == TokenType::Function) {
            if (sp < 1) return 0.0;
            double a = stack[--sp];
            int op = op_codes[i];
            double result;
            switch (op) {
                case 10: result = sin(a); break;
                case 11: result = cos(a); break;
                case 12: result = (a > 1e-15) ? log10(a) : -15.0; break;
                case 13: result = (a > 1e-15) ? log(a) : -34.5; break;
                case 14: result = exp(fmin(a, 709.0)); break;
                case 15: result = (a >= 0.0) ? sqrt(a) : 0.0; break;
                case 16: result = tan(a); break;
                case 17: result = fabs(a); break;
                default: result = 0.0;
            }
            stack[sp++] = result;
        }
    }
    return (sp > 0) ? stack[0] : 0.0;
}

__device__ __forceinline__ float eval_expr_device_fp16(
    const TokenType* types,
    const float* constants,
    const int* var_indices,
    const int* op_codes,
    int length,
    const float* vars)
{
    half stack[32];
    int sp = 0;
    
    for (int i = 0; i < length; ++i) {
        TokenType t = types[i];
        
        if (t == TokenType::Number) {
            stack[sp++] = __float2half(constants[i]);
        }
        else if (t == TokenType::Variable) {
            int idx = var_indices[i];
            stack[sp++] = (idx >= 0 && idx < 4) ? __float2half(vars[idx]) : __float2half(0.0f);
        }
        else if (t == TokenType::Operator) {
            if (sp < 2) return 0.0f;
            half b = stack[--sp];
            half a = stack[--sp];
            int op = op_codes[i];
            
            float af = __half2float(a);
            float bf = __half2float(b);
            float result;
            
            switch (op) {
                case 0: result = af + bf; break;
                case 1: result = af - bf; break;
                case 2: result = af * bf; break;
                case 3: result = (fabsf(bf) > 1e-5f) ? (af / bf) : 0.0f; break;
                case 4: result = powf(af, bf); break;
                default: result = 0.0f;
            }
            stack[sp++] = __float2half(result);
        }
        else if (t == TokenType::Function) {
            if (sp < 1) return 0.0f;
            half a = stack[--sp];
            int op = op_codes[i];
            
            float af = __half2float(a);
            float result;
            
            switch (op) {
                case 10: result = sinf(af); break;
                case 11: result = cosf(af); break;
                case 12: result = (af > 1e-5f) ? log10f(af) : -5.0f; break;
                case 13: result = (af > 1e-5f) ? logf(af) : -11.5f; break;
                case 14: result = expf(fminf(af, 11.0f)); break;
                case 15: result = (af >= 0.0f) ? sqrtf(af) : 0.0f; break;
                case 16: result = tanf(af); break;
                case 17: result = fabsf(af); break;
                default: result = 0.0f;
            }
            stack[sp++] = __float2half(result);
        }
    }
    
    return (sp > 0) ? __half2float(stack[0]) : 0.0f;
}

// ============================================================================
// FP32 KERNEL - NATIVE FLOAT ACCUMULATION (MAXIMUM SPEED)
// ============================================================================
__global__ void mc_integrate_kernel_fp32_optimized(
    const TokenType* __restrict__ d_types,
    const float* __restrict__ d_constants,
    const int* __restrict__ d_var_indices,
    const int* __restrict__ d_op_codes,
    int expr_length,
    const double* __restrict__ d_bounds_min,
    const double* __restrict__ d_bounds_max,
    int dimensions,
    size_t samples_per_term,
    int num_terms,
    float* __restrict__ d_results,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    // compute term and per-term block index to partition samples across multiple blocks
    int global_block = blockIdx.x;
    int grid_blocks = gridDim.x;
    int blocks_per_term = (grid_blocks > num_terms) ? (grid_blocks / num_terms) : 1;
    int term_id = global_block / blocks_per_term;
    int block_in_term = global_block % blocks_per_term;
    if (term_id >= num_terms) return;

    const int expr_offset = term_id * expr_length;
    unsigned long long rng_state = seed + ((unsigned long long)term_id << 32) + tid + (unsigned long long)block_in_term;
    
    // Compute volume in float (faster) and cache per-dimension bounds in registers
    float volume = 1.0f;
    float bmin_f[4];
    float delta_f[4];
    #pragma unroll
    for (int d = 0; d < dimensions; ++d) {
        double lo = d_bounds_min[d];
        double hi = d_bounds_max[d];
        bmin_f[d] = (float)lo;
        delta_f[d] = (float)(hi - lo);
        volume *= delta_f[d];
    }
    
    // *** NATIVE FLOAT ACCUMULATION - NO CONVERSION OVERHEAD ***
    float local_sum = 0.0f;
    
    const bool use_sobol = (d_sobol != nullptr);
    const size_t sobol_term_base = (size_t)term_id * samples_per_term * dimensions;
    size_t samples_per_block = (samples_per_term + blocks_per_term - 1) / blocks_per_term;
    size_t start = (size_t)block_in_term * samples_per_block;
    size_t end = start + samples_per_block;
    if (end > samples_per_term) end = samples_per_term;

    // Two specialized loops to avoid per-sample branching on Sobol vs RNG
    const float eps_f = 1e-12f;
    if (use_sobol) {
        for (size_t s = start + tid; s < end; s += block_size) {
            float vars[4];
            #pragma unroll
            for (int d = 0; d < dimensions; ++d) {
                float u = (float)d_sobol[sobol_term_base + s * dimensions + d];
                u = fminf(fmaxf(u, eps_f), 1.0f - eps_f);
                vars[d] = bmin_f[d] + u * delta_f[d];
            }
            float f_val = eval_expr_device_fp32(
                d_types + expr_offset,
                d_constants + expr_offset,
                d_var_indices + expr_offset,
                d_op_codes + expr_offset,
                expr_length,
                vars
            );
            local_sum += f_val;
        }
    } else {
        for (size_t s = start + tid; s < end; s += block_size) {
            float vars[4];
            #pragma unroll
            for (int d = 0; d < dimensions; ++d) {
                float u = lcg_random_float(rng_state);
                u = fminf(fmaxf(u, eps_f), 1.0f - eps_f);
                vars[d] = bmin_f[d] + u * delta_f[d];
            }
            float f_val = eval_expr_device_fp32(
                d_types + expr_offset,
                d_constants + expr_offset,
                d_var_indices + expr_offset,
                d_op_codes + expr_offset,
                expr_length,
                vars
            );
            local_sum += f_val;
        }
    }
    
    // Warp reduction with FLOAT (32-bit shuffle - 2x faster than double)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    
    __shared__ float warp_sums[8];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        const int num_warps = (block_size + 31) >> 5;
        float final_sum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
        }
        
        if (tid == 0) {
            float partial = (volume / (float)samples_per_term) * final_sum;
            atomicAdd(&d_results[term_id], partial);
        }
    }
}

// ============================================================================
// FP64 KERNEL - NATIVE DOUBLE ACCUMULATION (MAXIMUM ACCURACY)
// ============================================================================
__global__ void mc_integrate_kernel_fp64_optimized(
    const TokenType* __restrict__ d_types,
    const double* __restrict__ d_constants,
    const int* __restrict__ d_var_indices,
    const int* __restrict__ d_op_codes,
    int expr_length,
    const double* __restrict__ d_bounds_min,
    const double* __restrict__ d_bounds_max,
    int dimensions,
    size_t samples_per_term,
    int num_terms,
    double* __restrict__ d_results,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    // partition blocks among terms to increase grid size and occupancy
    int global_block = blockIdx.x;
    int term_id = global_block / 1; // default 1 block per term unless host requests more
    int block_in_term = 0;
    // If host passed larger grid (blocks = num_terms * blocks_per_term), recompute
    // blocks_per_term is encoded by caller via grid sizing logic; infer safe defaults here
    // If grid size > num_terms, derive blocks_per_term
    int grid_blocks = gridDim.x;
    if (grid_blocks > num_terms) {
        int blocks_per_term = grid_blocks / num_terms;
        term_id = global_block / blocks_per_term;
        block_in_term = global_block % blocks_per_term;
    }
    if (term_id >= num_terms) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int expr_offset = term_id * expr_length;

    unsigned long long rng_state = seed + ((unsigned long long)term_id << 32) + tid + (unsigned long long)block_in_term;

    double volume = 1.0;
    #pragma unroll
    for (int d = 0; d < dimensions; ++d) {
        volume *= (d_bounds_max[d] - d_bounds_min[d]);
    }

    // *** NATIVE DOUBLE ACCUMULATION ***
    double local_sum = 0.0;

    const bool use_sobol = (d_sobol != nullptr);
    const size_t sobol_term_base = (size_t)term_id * samples_per_term * dimensions;

    // compute per-block sample range
    int blocks_per_term = (gridDim.x > num_terms) ? (gridDim.x / num_terms) : 1;
    size_t samples_per_block = (samples_per_term + blocks_per_term - 1) / blocks_per_term;
    size_t start = (size_t)block_in_term * samples_per_block;
    size_t end = start + samples_per_block;
    if (end > samples_per_term) end = samples_per_term;

    // Avoid per-sample branching: separate Sobol and RNG loops
    const double eps_d = 1e-15;
    if (use_sobol) {
        for (size_t s = start + tid; s < end; s += block_size) {
            double vars[4];
            #pragma unroll
            for (int d = 0; d < dimensions; ++d) {
                double u = d_sobol[sobol_term_base + s * dimensions + d];
                u = fmin(fmax(u, eps_d), 1.0 - eps_d);
                vars[d] = d_bounds_min[d] + u * (d_bounds_max[d] - d_bounds_min[d]);
            }
            double f_val = eval_expr_device_fp64_native(
                d_types + expr_offset,
                d_constants + expr_offset,
                d_var_indices + expr_offset,
                d_op_codes + expr_offset,
                expr_length,
                vars
            );
            local_sum += f_val;
        }
    } else {
        for (size_t s = start + tid; s < end; s += block_size) {
            double vars[4];
            #pragma unroll
            for (int d = 0; d < dimensions; ++d) {
                double u = lcg_random_double(rng_state);
                u = fmin(fmax(u, eps_d), 1.0 - eps_d);
                vars[d] = d_bounds_min[d] + u * (d_bounds_max[d] - d_bounds_min[d]);
            }
            double f_val = eval_expr_device_fp64_native(
                d_types + expr_offset,
                d_constants + expr_offset,
                d_var_indices + expr_offset,
                d_op_codes + expr_offset,
                expr_length,
                vars
            );
            local_sum += f_val;
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    
    __shared__ double warp_sums[8];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        const int num_warps = (block_size + 31) >> 5;
        double final_sum = (tid < num_warps) ? warp_sums[tid] : 0.0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
        }
        
        if (tid == 0) {
            double partial = (volume / (double)samples_per_term) * final_sum;
            atomicAdd(&d_results[term_id], partial);
        }
    }
}

// ============================================================================
// FP16 KERNEL - NATIVE FLOAT ACCUMULATION (MAXIMUM SPEED, GOOD ENOUGH ACCURACY)
// ============================================================================
__global__ void mc_integrate_kernel_fp16_optimized(
    const TokenType* __restrict__ d_types,
    const float* __restrict__ d_constants,
    const int* __restrict__ d_var_indices,
    const int* __restrict__ d_op_codes,
    int expr_length,
    const double* __restrict__ d_bounds_min,
    const double* __restrict__ d_bounds_max,
    int dimensions,
    size_t samples_per_term,
    int num_terms,
    float* __restrict__ d_results,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    const int term_id = blockIdx.x;
    if (term_id >= num_terms) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int expr_offset = term_id * expr_length;
    
    unsigned long long rng_state = seed + ((unsigned long long)term_id << 32) + tid;
    
    // Cache float bounds and volume once
    float volume = 1.0f;
    float bmin_f[4];
    float delta_f[4];
    #pragma unroll
    for (int d = 0; d < dimensions; ++d) {
        bmin_f[d] = (float)d_bounds_min[d];
        delta_f[d] = (float)(d_bounds_max[d] - d_bounds_min[d]);
        volume *= delta_f[d];
    }

    // *** NATIVE FLOAT ACCUMULATION - FAST AND SUFFICIENT FOR FP16 ***
    float local_sum = 0.0f;

    const bool use_sobol = (d_sobol != nullptr);
    const size_t sobol_term_base = (size_t)term_id * samples_per_term * dimensions;
    const float eps_f = 1e-12f;

    if (use_sobol) {
        for (size_t s = tid; s < samples_per_term; s += block_size) {
            float vars[4];
            #pragma unroll
            for (int d = 0; d < dimensions; ++d) {
                float u = (float)d_sobol[sobol_term_base + s * dimensions + d];
                u = fminf(fmaxf(u, eps_f), 1.0f - eps_f);
                vars[d] = bmin_f[d] + u * delta_f[d];
            }
            float f_val = eval_expr_device_fp16(
                d_types + expr_offset,
                d_constants + expr_offset,
                d_var_indices + expr_offset,
                d_op_codes + expr_offset,
                expr_length,
                vars
            );
            local_sum += f_val;
        }
    } else {
        for (size_t s = tid; s < samples_per_term; s += block_size) {
            float vars[4];
            #pragma unroll
            for (int d = 0; d < dimensions; ++d) {
                float u = lcg_random_float(rng_state);
                u = fminf(fmaxf(u, eps_f), 1.0f - eps_f);
                vars[d] = bmin_f[d] + u * delta_f[d];
            }
            float f_val = eval_expr_device_fp16(
                d_types + expr_offset,
                d_constants + expr_offset,
                d_var_indices + expr_offset,
                d_op_codes + expr_offset,
                expr_length,
                vars
            );
            local_sum += f_val;
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    
    __shared__ float warp_sums[8];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        const int num_warps = (block_size + 31) >> 5;
        float final_sum = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
        }
        
        if (tid == 0) {
            d_results[term_id] = (volume / (float)samples_per_term) * final_sum;
        }
    }
}

// ============================================================================
// MIXED PRECISION KERNEL - PRECISION-SPECIFIC ACCUMULATION
// ============================================================================
__global__ void mc_integrate_kernel_mixed_optimized(
    const TokenType* __restrict__ d_types,
    const float* __restrict__ d_constants,
    const int* __restrict__ d_var_indices,
    const int* __restrict__ d_op_codes,
    int expr_length,
    const double* __restrict__ d_bounds_min_per_term,
    const double* __restrict__ d_bounds_max_per_term,
    int dimensions,
    const int* __restrict__ d_samples_per_term,
    const unsigned long long* __restrict__ d_sample_offsets,
    unsigned long long total_samples,
    int num_terms,
    double* __restrict__ d_results,
    const int* __restrict__ d_precisions,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    const int term_id = blockIdx.x;
    if (term_id >= num_terms) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const int prec = d_precisions[term_id];
    const int expr_offset = term_id * expr_length;
    const size_t samples_this_term = d_samples_per_term[term_id];
    
    const double* bounds_min = d_bounds_min_per_term + (size_t)term_id * dimensions;
    const double* bounds_max = d_bounds_max_per_term + (size_t)term_id * dimensions;
    
    unsigned long long rng_state = seed + ((unsigned long long)term_id << 32) + tid;
    
    const bool use_sobol = (d_sobol != nullptr);
    const size_t sobol_offset = use_sobol ? d_sample_offsets[term_id] : 0;
    
    double result_d = 0.0;
    
    // *** PRECISION-SPECIFIC PATHS - NO UNNECESSARY CONVERSIONS ***
    if (prec == 2) {
        // FP64 path: double accumulation
        double volume = 1.0;
        for (int d = 0; d < dimensions; ++d) {
            volume *= (bounds_max[d] - bounds_min[d]);
        }
        
        double local_sum = 0.0;
        
        for (size_t s = tid; s < samples_this_term; s += block_size) {
            double vars[4];
            
            for (int d = 0; d < dimensions; ++d) {
                double u = use_sobol ? d_sobol[(sobol_offset + s) * dimensions + d] 
                                     : lcg_random_double(rng_state);
                vars[d] = bounds_min[d] + u * (bounds_max[d] - bounds_min[d]);
            }
            
            local_sum += eval_expr_device_fp64(
                d_types + expr_offset, d_constants + expr_offset,
                d_var_indices + expr_offset, d_op_codes + expr_offset,
                expr_length, vars);
        }
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        
        __shared__ double warp_sums_d[8];
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        
        if (lane_id == 0) warp_sums_d[warp_id] = local_sum;
        __syncthreads();
        
        if (warp_id == 0) {
            double final_sum = (tid < (block_size >> 5)) ? warp_sums_d[tid] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
            }
            if (tid == 0) {
                result_d = (volume / (double)samples_this_term) * final_sum;
            }
        }
        
    } else {
        // FP32/FP16 path: float accumulation for SPEED
        float volume = 1.0f;
        for (int d = 0; d < dimensions; ++d) {
            volume *= (float)(bounds_max[d] - bounds_min[d]);
        }
        
        float local_sum = 0.0f;
        
        for (size_t s = tid; s < samples_this_term; s += block_size) {
            float vars[4];
            
            for (int d = 0; d < dimensions; ++d) {
                float u = use_sobol ? (float)d_sobol[(sobol_offset + s) * dimensions + d]
                                    : lcg_random_float(rng_state);
                vars[d] = (float)bounds_min[d] + u * (float)(bounds_max[d] - bounds_min[d]);
            }
            
            if (prec == 1) {
                local_sum += eval_expr_device_fp32(
                    d_types + expr_offset, d_constants + expr_offset,
                    d_var_indices + expr_offset, d_op_codes + expr_offset,
                    expr_length, vars);
            } else {
                local_sum += eval_expr_device_fp16(
                    d_types + expr_offset, d_constants + expr_offset,
                    d_var_indices + expr_offset, d_op_codes + expr_offset,
                    expr_length, vars);
            }
        }
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        
        __shared__ float warp_sums_f[8];
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        
        if (lane_id == 0) warp_sums_f[warp_id] = local_sum;
        __syncthreads();
        
        if (warp_id == 0) {
            float final_sum = (tid < (block_size >> 5)) ? warp_sums_f[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
            }
            if (tid == 0) {
                result_d = (double)((volume / (float)samples_this_term) * final_sum);
            }
        }
    }
    
    if (tid == 0) {
        // accumulate partial results from multiple blocks
        atomicAdd(&d_results[term_id], result_d);
    }
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

void prepare_compiled_cuda_data(
    const CompiledExpr& compiled,
    TokenType** d_types,
    float** d_constants,
    int** d_var_indices,
    int** d_op_codes,
    int* expr_length)
{
    *expr_length = compiled.expr_length;
    size_t size = compiled.expr_length;
    
    CUDA_CHECK(cudaMalloc(d_types, size * sizeof(TokenType)));
    CUDA_CHECK(cudaMalloc(d_constants, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_var_indices, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_op_codes, size * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(*d_types, compiled.types.data(), 
                          size * sizeof(TokenType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_constants, compiled.constants.data(), 
                          size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_var_indices, compiled.var_indices.data(), 
                          size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_op_codes, compiled.op_codes.data(), 
                          size * sizeof(int), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> monte_carlo_integrate_nd_cuda_batch(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& all_compiled,
    const GPUConfig& config)
{
    int num_terms = static_cast<int>(all_compiled.size());
    int dimensions = static_cast<int>(bounds_min.size());

    if (num_terms == 0 || dimensions == 0 || samples == 0)
        return std::vector<T>(num_terms, T(0));

    // Bounds
    double *d_bounds_min, *d_bounds_max;
    CUDA_CHECK(cudaMalloc(&d_bounds_min, dimensions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bounds_max, dimensions * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_bounds_min, bounds_min.data(),
                          dimensions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds_max, bounds_max.data(),
                          dimensions * sizeof(double), cudaMemcpyHostToDevice));

    // Expressions
    int expr_length = 0;
    for (const auto& c : all_compiled)
        expr_length = std::max(expr_length, c.expr_length);

    size_t total_expr_size = (size_t)num_terms * expr_length;

    std::vector<TokenType> all_types(total_expr_size, TokenType::LeftParen);
    std::vector<float> all_constants(total_expr_size, 0.0f);
    std::vector<int> all_var_indices(total_expr_size, -1);
    std::vector<int> all_op_codes(total_expr_size, -1);

    for (int i = 0; i < num_terms; ++i) {
        int offset = i * expr_length;
        const auto& c = all_compiled[i];
        for (int j = 0; j < c.expr_length; ++j) {
            all_types[offset + j] = c.types[j];
            all_constants[offset + j] = c.constants[j];
            all_var_indices[offset + j] = c.var_indices[j];
            all_op_codes[offset + j] = c.op_codes[j];
        }
    }

    TokenType *d_types;
    float *d_constants;
    int *d_var_indices, *d_op_codes;

    CUDA_CHECK(cudaMalloc(&d_types, total_expr_size * sizeof(TokenType)));
    CUDA_CHECK(cudaMalloc(&d_constants, total_expr_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var_indices, total_expr_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_op_codes, total_expr_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_types, all_types.data(),
                          total_expr_size * sizeof(TokenType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_constants, all_constants.data(),
                          total_expr_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_var_indices, all_var_indices.data(),
                          total_expr_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_op_codes, all_op_codes.data(),
                          total_expr_size * sizeof(int), cudaMemcpyHostToDevice));

    // Sobol sequence
    double* d_sobol = nullptr;
    const size_t SOBOL_THRESHOLD = 50'000'000ULL;

    size_t sobol_elems = (size_t)num_terms * samples * dimensions;
    if (sobol_elems <= SOBOL_THRESHOLD) {
        if (cudaMalloc(&d_sobol, sobol_elems * sizeof(double)) == cudaSuccess) {
            curandGenerator_t gen;
            if (curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64) == CURAND_STATUS_SUCCESS) {
                if (curandGenerateUniformDouble(gen, d_sobol, sobol_elems) == CURAND_STATUS_SUCCESS) {
                    // Success
                } else {
                    cudaFree(d_sobol);
                    d_sobol = nullptr;
                }
                curandDestroyGenerator(gen);
            } else {
                cudaFree(d_sobol);
                d_sobol = nullptr;
            }
        }
    }

    // AGGRESSIVE LAUNCH CONFIGURATION FOR MAXIMUM THROUGHPUT
    int optimal_threads = 256;
    int blocks_per_term = 1;
    
    // For very large sample counts, use multiple blocks per term
    if (samples > 10'000'000) {
        blocks_per_term = 4;  // Better load balancing
    }
    
    dim3 blocks(num_terms * blocks_per_term);
    dim3 threads(optimal_threads);

    static unsigned long long base_seed = 0x123456789abcdefULL;
    base_seed += 0x9e3779b97f4a7c15ULL;

    std::vector<T> host_results(num_terms);

    if constexpr (std::is_same<T, double>::value) {
        double* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, num_terms * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_results, 0, num_terms * sizeof(double)));

        // Prepare native double constants buffer for true FP64 evaluation
        double* d_constants_d = nullptr;
        std::vector<double> all_constants_d(total_expr_size);
        for (size_t i = 0; i < total_expr_size; ++i) all_constants_d[i] = (double)all_constants[i];
        CUDA_CHECK(cudaMalloc(&d_constants_d, total_expr_size * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_constants_d, all_constants_d.data(),
                              total_expr_size * sizeof(double), cudaMemcpyHostToDevice));

        mc_integrate_kernel_fp64_optimized<<<blocks, threads>>>(
            d_types, d_constants_d, d_var_indices, d_op_codes,
            expr_length, d_bounds_min, d_bounds_max,
            dimensions, samples, num_terms,
            d_results, base_seed, d_sobol);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(host_results.data(), d_results,
                              num_terms * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_constants_d));

    } else if constexpr (std::is_same<T, float>::value) {
        float* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, num_terms * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_results, 0, num_terms * sizeof(float)));

        mc_integrate_kernel_fp32_optimized<<<blocks, threads>>>(
            d_types, d_constants, d_var_indices, d_op_codes,
            expr_length, d_bounds_min, d_bounds_max,
            dimensions, samples, num_terms,
            d_results, base_seed, d_sobol);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> tmp(num_terms);
        CUDA_CHECK(cudaMemcpy(tmp.data(), d_results,
                              num_terms * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_terms; ++i)
            host_results[i] = static_cast<T>(tmp[i]);

        CUDA_CHECK(cudaFree(d_results));
    }

    CUDA_CHECK(cudaFree(d_types));
    CUDA_CHECK(cudaFree(d_constants));
    CUDA_CHECK(cudaFree(d_var_indices));
    CUDA_CHECK(cudaFree(d_op_codes));
    CUDA_CHECK(cudaFree(d_bounds_min));
    CUDA_CHECK(cudaFree(d_bounds_max));
    if (d_sobol) CUDA_CHECK(cudaFree(d_sobol));

    return host_results;
}

std::vector<float> monte_carlo_integrate_nd_cuda_batch_fp16(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& all_compiled,
    const GPUConfig& config)
{
    int num_terms = static_cast<int>(all_compiled.size());
    int dimensions = static_cast<int>(bounds_min.size());

    if (num_terms == 0 || dimensions == 0 || samples == 0)
        return std::vector<float>(num_terms, 0.0f);

    double *d_bounds_min, *d_bounds_max;
    CUDA_CHECK(cudaMalloc(&d_bounds_min, dimensions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bounds_max, dimensions * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_bounds_min, bounds_min.data(),
                          dimensions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds_max, bounds_max.data(),
                          dimensions * sizeof(double), cudaMemcpyHostToDevice));

    int expr_length = 0;
    for (const auto& c : all_compiled)
        expr_length = std::max(expr_length, c.expr_length);

    size_t total_expr_size = (size_t)num_terms * expr_length;

    std::vector<TokenType> all_types(total_expr_size, TokenType::LeftParen);
    std::vector<float> all_constants(total_expr_size, 0.0f);
    std::vector<int> all_var_indices(total_expr_size, -1);
    std::vector<int> all_op_codes(total_expr_size, -1);

    for (int i = 0; i < num_terms; ++i) {
        int offset = i * expr_length;
        const auto& c = all_compiled[i];
        for (int j = 0; j < c.expr_length; ++j) {
            all_types[offset + j] = c.types[j];
            all_constants[offset + j] = c.constants[j];
            all_var_indices[offset + j] = c.var_indices[j];
            all_op_codes[offset + j] = c.op_codes[j];
        }
    }

    TokenType *d_types;
    float *d_constants;
    int *d_var_indices, *d_op_codes;

    CUDA_CHECK(cudaMalloc(&d_types, total_expr_size * sizeof(TokenType)));
    CUDA_CHECK(cudaMalloc(&d_constants, total_expr_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var_indices, total_expr_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_op_codes, total_expr_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_types, all_types.data(),
                          total_expr_size * sizeof(TokenType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_constants, all_constants.data(),
                          total_expr_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_var_indices, all_var_indices.data(),
                          total_expr_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_op_codes, all_op_codes.data(),
                          total_expr_size * sizeof(int), cudaMemcpyHostToDevice));

    float* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_terms * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_results, 0, num_terms * sizeof(float)));

    dim3 blocks(num_terms);
    dim3 threads(config.threads_per_block);

    double* d_sobol = nullptr;
    const size_t SOBOL_LIMIT = 20'000'000ULL;

    size_t sobol_elems = (size_t)num_terms * samples * dimensions;
    if (sobol_elems <= SOBOL_LIMIT) {
        if (cudaMalloc(&d_sobol, sobol_elems * sizeof(double)) == cudaSuccess) {
            curandGenerator_t gen;
            if (curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64) == CURAND_STATUS_SUCCESS) {
                if (curandGenerateUniformDouble(gen, d_sobol, sobol_elems) != CURAND_STATUS_SUCCESS) {
                    cudaFree(d_sobol);
                    d_sobol = nullptr;
                }
                curandDestroyGenerator(gen);
            } else {
                cudaFree(d_sobol);
                d_sobol = nullptr;
            }
        }
    }

    static unsigned long long seed = 0xdeadbeefcafebabeULL;
    seed += 0x9e3779b97f4a7c15ULL;

    mc_integrate_kernel_fp16_optimized<<<blocks, threads>>>(
        d_types, d_constants, d_var_indices, d_op_codes,
        expr_length, d_bounds_min, d_bounds_max,
        dimensions, samples, num_terms,
        d_results, seed, d_sobol);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> results(num_terms);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results,
                          num_terms * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_types));
    CUDA_CHECK(cudaFree(d_constants));
    CUDA_CHECK(cudaFree(d_var_indices));
    CUDA_CHECK(cudaFree(d_op_codes));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_bounds_min));
    CUDA_CHECK(cudaFree(d_bounds_max));
    if (d_sobol) CUDA_CHECK(cudaFree(d_sobol));

    return results;
}

// --- v2-style region batch kernel and host wrapper (ported and adapted)
#define MAX_DIMS 8

template <typename T>
__global__ void monte_carlo_regions_compiled_kernel(
    TokenType* types,
    float* constants,
    int* var_indices,
    int* op_codes,
    int expr_length,
    T* region_lower_bounds,
    T* region_upper_bounds,
    int* region_dims_array,
    int num_regions,
    unsigned long long samples_per_thread,
    T* d_results,
    unsigned long long* d_valid_counts,
    unsigned long long seed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int region_idx = blockIdx.y;
    if (region_idx >= num_regions) return;

    int dims = region_dims_array[region_idx];
    if (dims > MAX_DIMS) return;

    curandState state;
    curand_init(seed + tid + region_idx * 10000, 0, 0, &state);

    T sum = 0;
    unsigned long long valid = 0;

    T point[MAX_DIMS];
    T* lower = region_lower_bounds + region_idx * MAX_DIMS;
    T* upper = region_upper_bounds + region_idx * MAX_DIMS;

    for (unsigned long long i = 0; i < samples_per_thread; ++i) {
        // generate random point uniform in region
        for (int d = 0; d < dims; ++d) {
            float r = curand_uniform(&state);
            point[d] = lower[d] + (upper[d] - lower[d]) * static_cast<T>(r);
        }

            // Evaluate point using appropriate device evaluator
        T value = 0;
        if (sizeof(T) == sizeof(float)) {
            value = static_cast<T>(eval_expr_device_fp32(
                types, constants, var_indices, op_codes, expr_length, reinterpret_cast<const float*>(point)));
        } else {
            value = static_cast<T>(eval_expr_device_fp64(
                types, constants, var_indices, op_codes, expr_length, reinterpret_cast<const double*>(point)));
        }

        if (isfinite(value)) {
            sum += value;
            valid++;
        }
    }

    const int grid_stride = gridDim.x * blockDim.x;
    d_results[region_idx * grid_stride + tid] = sum;
    d_valid_counts[region_idx * grid_stride + tid] = valid;
}

template <typename T>
std::vector<T> monte_carlo_integrate_regions_cuda_batch_compiled(
    const std::vector<Region>& regions,
    size_t samples_per_region,
    const CompiledExpr& compiled)
{
    int num_regions = static_cast<int>(regions.size());
    if (num_regions == 0) return {};

    const int threadsPerBlock = 256;
    const int blocksPerRegion = 32;

    dim3 grid(blocksPerRegion, num_regions);
    dim3 block(threadsPerBlock);

    unsigned long long samples_per_thread = 
        (samples_per_region + blocksPerRegion * threadsPerBlock - 1) / 
        (blocksPerRegion * threadsPerBlock);

    // Prepare compiled expression
    TokenType* d_types;
    float* d_constants;
    int* d_var_indices;
    int* d_op_codes;
    int expr_length;

    prepare_compiled_cuda_data(compiled, &d_types, &d_constants,
                              &d_var_indices, &d_op_codes, &expr_length);

    // Prepare region bounds (padded to MAX_DIMS)
    std::vector<T> h_lower_bounds(num_regions * MAX_DIMS, 0);
    std::vector<T> h_upper_bounds(num_regions * MAX_DIMS, 0);
    std::vector<int> h_region_dims(num_regions);

    for (int i = 0; i < num_regions; i++) {
        int dims = static_cast<int>(regions[i].bounds_min.size());
        h_region_dims[i] = dims;

        for (int d = 0; d < dims; d++) {
            h_lower_bounds[i * MAX_DIMS + d] = static_cast<T>(regions[i].bounds_min[d]);
            h_upper_bounds[i * MAX_DIMS + d] = static_cast<T>(regions[i].bounds_max[d]);
        }
    }

    T* d_lower_bounds;
    T* d_upper_bounds;
    int* d_region_dims;

    cudaMalloc(&d_lower_bounds, num_regions * MAX_DIMS * sizeof(T));
    cudaMalloc(&d_upper_bounds, num_regions * MAX_DIMS * sizeof(T));
    cudaMalloc(&d_region_dims, num_regions * sizeof(int));

    cudaMemcpy(d_lower_bounds, h_lower_bounds.data(), 
               num_regions * MAX_DIMS * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper_bounds, h_upper_bounds.data(), 
               num_regions * MAX_DIMS * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_region_dims, h_region_dims.data(), 
               num_regions * sizeof(int), cudaMemcpyHostToDevice);

    const size_t results_size = grid.x * grid.y * block.x;
    T* d_results;
    unsigned long long* d_valid_counts;
    cudaMalloc(&d_results, results_size * sizeof(T));
    cudaMalloc(&d_valid_counts, results_size * sizeof(unsigned long long));

    unsigned long long seed = std::chrono::high_resolution_clock::now()
                                .time_since_epoch().count();

    monte_carlo_regions_compiled_kernel<T><<<grid, block>>>(
        d_types, d_constants, d_var_indices, d_op_codes, expr_length,
        d_lower_bounds, d_upper_bounds, d_region_dims, num_regions,
        samples_per_thread, d_results, d_valid_counts, seed);

    std::vector<T> h_results(results_size);
    std::vector<unsigned long long> h_valid_counts(results_size);

    cudaMemcpy(h_results.data(), d_results, results_size * sizeof(T), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid_counts.data(), d_valid_counts, 
               results_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::vector<T> final_results(num_regions);
    const int points_per_region = grid.x * block.x;

    for (int region = 0; region < num_regions; region++) {
        T sum = 0;
        unsigned long long valid_total = 0;

        for (int i = 0; i < points_per_region; i++) {
            const int idx = region * points_per_region + i;
            sum += h_results[idx];
            valid_total += h_valid_counts[idx];
        }

        if (valid_total > 0) {
            T volume = static_cast<T>(regions[region].volume());
            final_results[region] = volume * sum / valid_total;
        } else {
            final_results[region] = 0;
        }
    }

    cudaFree(d_types);
    cudaFree(d_constants);
    cudaFree(d_var_indices);
    cudaFree(d_op_codes);
    cudaFree(d_lower_bounds);
    cudaFree(d_upper_bounds);
    cudaFree(d_region_dims);
    cudaFree(d_results);
    cudaFree(d_valid_counts);

    return final_results;
}

template <typename T>
std::vector<T> monte_carlo_integrate_regions_cuda_adaptive(
    const std::vector<Region>& regions,
    size_t samples_per_region,
    const CompiledExpr& compiled,
    const GPUConfig& config)
{
    int num_regions = static_cast<int>(regions.size());
    if (num_regions == 0 || samples_per_region == 0)
        return std::vector<T>();

    int dimensions = static_cast<int>(regions[0].bounds_min.size());

    std::vector<std::vector<double>> bounds_min_per_term(num_regions);
    std::vector<std::vector<double>> bounds_max_per_term(num_regions);
    std::vector<size_t> samples_per_term(num_regions, samples_per_region);
    std::vector<CompiledExpr> exprs(num_regions, compiled);
    std::vector<Precision> precisions(num_regions);

    for (int i = 0; i < num_regions; ++i) {
        bounds_min_per_term[i] = regions[i].bounds_min;
        bounds_max_per_term[i] = regions[i].bounds_max;
        precisions[i] = std::is_same<T, double>::value ? Precision::Double : Precision::Float;
    }

    auto res = monte_carlo_integrate_nd_cuda_batch_mixed(
        samples_per_region,
        regions[0].bounds_min,
        regions[0].bounds_max,
        bounds_min_per_term,
        bounds_max_per_term,
        samples_per_term,
        exprs,
        precisions,
        config);

    std::vector<T> out(num_regions);
    for (int i = 0; i < num_regions; ++i) out[i] = static_cast<T>(res[i]);

    return out;
}

std::vector<double> monte_carlo_integrate_nd_cuda_batch_mixed(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<std::vector<double>>& bounds_min_per_term,
    const std::vector<std::vector<double>>& bounds_max_per_term,
    const std::vector<size_t>& samples_per_term,
    const std::vector<CompiledExpr>& all_compiled,
    const std::vector<Precision>& precisions,
    const GPUConfig& config)
{
    int num_terms = static_cast<int>(all_compiled.size());
    if (num_terms == 0) return {};

    int dimensions = !bounds_min_per_term.empty()
                     ? static_cast<int>(bounds_min_per_term[0].size())
                     : static_cast<int>(bounds_min.size());
    if (dimensions == 0) return std::vector<double>(num_terms, 0.0);

    int expr_length = 0;
    for (const auto& c : all_compiled)
        expr_length = std::max(expr_length, c.expr_length);

    size_t total_expr_size = static_cast<size_t>(num_terms) * expr_length;

    std::vector<TokenType> all_types(total_expr_size, TokenType::LeftParen);
    std::vector<float> all_constants(total_expr_size, 0.0f);
    std::vector<int> all_var_indices(total_expr_size, -1);
    std::vector<int> all_op_codes(total_expr_size, -1);

    for (int i = 0; i < num_terms; ++i) {
        int offset = i * expr_length;
        for (int j = 0; j < all_compiled[i].expr_length; ++j) {
            all_types[offset + j] = all_compiled[i].types[j];
            all_constants[offset + j] = all_compiled[i].constants[j];
            all_var_indices[offset + j] = all_compiled[i].var_indices[j];
            all_op_codes[offset + j] = all_compiled[i].op_codes[j];
        }
    }

    TokenType *d_types;
    float *d_constants;
    int *d_var_indices, *d_op_codes;

    CUDA_CHECK(cudaMalloc(&d_types, total_expr_size * sizeof(TokenType)));
    CUDA_CHECK(cudaMalloc(&d_constants, total_expr_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var_indices, total_expr_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_op_codes, total_expr_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_types, all_types.data(), 
                          total_expr_size * sizeof(TokenType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_constants, all_constants.data(), 
                          total_expr_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_var_indices, all_var_indices.data(), 
                          total_expr_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_op_codes, all_op_codes.data(), 
                          total_expr_size * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<double> flat_min(num_terms * dimensions);
    std::vector<double> flat_max(num_terms * dimensions);

    for (int i = 0; i < num_terms; ++i) {
        for (int d = 0; d < dimensions; ++d) {
            flat_min[i * dimensions + d] = 
                bounds_min_per_term.empty() ? bounds_min[d] : bounds_min_per_term[i][d];
            flat_max[i * dimensions + d] = 
                bounds_max_per_term.empty() ? bounds_max[d] : bounds_max_per_term[i][d];
        }
    }

    double *d_bounds_min, *d_bounds_max;
    CUDA_CHECK(cudaMalloc(&d_bounds_min, flat_min.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bounds_max, flat_max.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_bounds_min, flat_min.data(), 
                          flat_min.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds_max, flat_max.data(), 
                          flat_max.size() * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<int> prec_i(num_terms);
    std::vector<int> samp_i(num_terms);
    for (int i = 0; i < num_terms; ++i) {
        prec_i[i] = static_cast<int>(precisions[i]);
        samp_i[i] = static_cast<int>(samples_per_term[i]);
    }

    int *d_precisions, *d_samples;
    CUDA_CHECK(cudaMalloc(&d_precisions, num_terms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_samples, num_terms * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_precisions, prec_i.data(), 
                          num_terms * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_samples, samp_i.data(), 
                          num_terms * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<unsigned long long> offsets(num_terms);
    unsigned long long total_samples = 0;
    for (int i = 0; i < num_terms; ++i) {
        offsets[i] = total_samples;
        total_samples += static_cast<unsigned long long>(samp_i[i]);
    }

    unsigned long long* d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, num_terms * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), 
                          num_terms * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    double* d_sobol = nullptr;
    const size_t SOBOL_LIMIT = 30'000'000ULL;

    if (total_samples * dimensions <= SOBOL_LIMIT) {
        if (cudaMalloc(&d_sobol, total_samples * dimensions * sizeof(double)) == cudaSuccess) {
            curandGenerator_t gen;
            if (curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64) == CURAND_STATUS_SUCCESS) {
                if (curandGenerateUniformDouble(gen, d_sobol, 
                    total_samples * dimensions) != CURAND_STATUS_SUCCESS) {
                    cudaFree(d_sobol);
                    d_sobol = nullptr;
                }
                curandDestroyGenerator(gen);
            } else {
                cudaFree(d_sobol);
                d_sobol = nullptr;
            }
        }
    }

    double* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_terms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_results, 0, num_terms * sizeof(double)));

    dim3 blocks(num_terms);
    dim3 threads(config.threads_per_block);

    static unsigned long long seed = 0xdeadbeef12345678ULL;
    seed += 0x9e3779b97f4a7c15ULL;

    mc_integrate_kernel_mixed_optimized<<<blocks, threads>>>(
        d_types, d_constants, d_var_indices, d_op_codes, expr_length,
        d_bounds_min, d_bounds_max, dimensions,
        d_samples, d_offsets, total_samples, num_terms,
        d_results, d_precisions, seed, d_sobol);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> results(num_terms);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, 
                          num_terms * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_types);
    cudaFree(d_constants);
    cudaFree(d_var_indices);
    cudaFree(d_op_codes);
    cudaFree(d_bounds_min);
    cudaFree(d_bounds_max);
    cudaFree(d_precisions);
    cudaFree(d_samples);
    cudaFree(d_offsets);
    cudaFree(d_results);
    if (d_sobol) cudaFree(d_sobol);

    return results;
}

// Explicit template instantiations
template std::vector<float> monte_carlo_integrate_nd_cuda_batch<float>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);

template std::vector<double> monte_carlo_integrate_nd_cuda_batch<double>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);

template std::vector<float> monte_carlo_integrate_regions_cuda_adaptive<float>(
    const std::vector<Region>&, size_t, const CompiledExpr&, const GPUConfig&);

template std::vector<double> monte_carlo_integrate_regions_cuda_adaptive<double>(
    const std::vector<Region>&, size_t, const CompiledExpr&, const GPUConfig&);