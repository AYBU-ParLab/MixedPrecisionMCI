#include "cuda_integration.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cmath>

// Constant memory for frequently used FP16 constants (reduces register pressure)
__constant__ half c_fp16_zero;
__constant__ half c_fp16_one;
__constant__ half c_fp16_half;
__constant__ half c_fp16_eps;

// Initialize constant memory on host
void init_fp16_constants() {
    half h_zero = __float2half(0.0f);
    half h_one = __float2half(1.0f);
    half h_half = __float2half(0.5f);
    half h_eps = __float2half(1e-5f);

    cudaMemcpyToSymbol(c_fp16_zero, &h_zero, sizeof(half));
    cudaMemcpyToSymbol(c_fp16_one, &h_one, sizeof(half));
    cudaMemcpyToSymbol(c_fp16_half, &h_half, sizeof(half));
    cudaMemcpyToSymbol(c_fp16_eps, &h_eps, sizeof(half));
}

GPUConfig detect_gpu() {
    GPUConfig config;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    config.sm_count = prop.multiProcessorCount;
    // RTX 4050 (Ada Lovelace, sm_89) benefits from 512 threads per block
    // Better occupancy and warp scheduling compared to 256
    config.threads_per_block = 512;
    config.blocks_per_sm = 2;  // Adjust blocks_per_sm to maintain total thread count

    return config;
}

// Xorshift64* - Faster and better quality than LCG
__device__ __forceinline__ double xorshift_random_double(unsigned long long& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (double)((state * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ float xorshift_random_float(unsigned long long& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (float)(((state * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0));
}

// Vectorized RNG for reduced overhead
__device__ __forceinline__ float2 xorshift_random_float2(unsigned long long& state) {
    // First random number
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t1 = state * 0x2545F4914F6CDD1DULL;
    float x = (float)((t1 >> 11) * (1.0 / 9007199254740992.0));

    // Second random number
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t2 = state * 0x2545F4914F6CDD1DULL;
    float y = (float)((t2 >> 11) * (1.0 / 9007199254740992.0));

    return make_float2(x, y);
}

__device__ __forceinline__ float4 xorshift_random_float4(unsigned long long& state) {
    // Generate 4 random numbers efficiently
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t1 = state * 0x2545F4914F6CDD1DULL;
    float x = (float)((t1 >> 11) * (1.0 / 9007199254740992.0));

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t2 = state * 0x2545F4914F6CDD1DULL;
    float y = (float)((t2 >> 11) * (1.0 / 9007199254740992.0));

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t3 = state * 0x2545F4914F6CDD1DULL;
    float z = (float)((t3 >> 11) * (1.0 / 9007199254740992.0));

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t4 = state * 0x2545F4914F6CDD1DULL;
    float w = (float)((t4 >> 11) * (1.0 / 9007199254740992.0));

    return make_float4(x, y, z, w);
}

// Double precision vectorized RNG
__device__ __forceinline__ double2 xorshift_random_double2(unsigned long long& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t1 = state * 0x2545F4914F6CDD1DULL;
    double x = (double)((t1 >> 11) * (1.0 / 9007199254740992.0));

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t2 = state * 0x2545F4914F6CDD1DULL;
    double y = (double)((t2 >> 11) * (1.0 / 9007199254740992.0));

    return make_double2(x, y);
}

__device__ __forceinline__ double4 xorshift_random_double4(unsigned long long& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t1 = state * 0x2545F4914F6CDD1DULL;
    double x = (double)((t1 >> 11) * (1.0 / 9007199254740992.0));

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t2 = state * 0x2545F4914F6CDD1DULL;
    double y = (double)((t2 >> 11) * (1.0 / 9007199254740992.0));

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t3 = state * 0x2545F4914F6CDD1DULL;
    double z = (double)((t3 >> 11) * (1.0 / 9007199254740992.0));

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    unsigned long long t4 = state * 0x2545F4914F6CDD1DULL;
    double w = (double)((t4 >> 11) * (1.0 / 9007199254740992.0));

    return make_double4(x, y, z, w);
}

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
            stack[sp++] = (idx >= 0 && idx < 10) ? vars[idx] : 0.0f;
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
            stack[sp++] = (idx >= 0 && idx < 10) ? vars[idx] : 0.0;
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
            stack[sp++] = (idx >= 0 && idx < 10) ? vars[idx] : 0.0;
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
            stack[sp++] = (idx >= 0 && idx < 10) ? __float2half(vars[idx]) : __float2half(0.0f);
        }
        else if (t == TokenType::Operator) {
            if (sp < 2) return 0.0f;
            half b = stack[--sp];
            half a = stack[--sp];
            int op = op_codes[i];

            half result;
            switch (op) {
                case 0: result = __hadd(a, b); break;                    // Native FP16 add
                case 1: result = __hsub(a, b); break;                    // Native FP16 sub
                case 2: result = __hmul(a, b); break;                    // Native FP16 mul
                case 3: {                                                 // Division
                    result = (__hgt(__habs(b), c_fp16_eps)) ? __hdiv(a, b) : c_fp16_zero;
                    break;
                }
                case 4: {                                                 // Power: a^b
                    // Try to detect common cases and use native FP16 operations

                    // Special case: x^0.5 = sqrt(x) - fully FP16
                    if (__heq(b, c_fp16_half)) {
                        #if __CUDA_ARCH__ >= 530
                        result = __hgt(a, c_fp16_zero) ? hsqrt(a) : c_fp16_zero;
                        #else
                        float af = __half2float(a);
                        result = __float2half((af >= 0.0f) ? sqrtf(af) : 0.0f);
                        #endif
                    }
                    // Check for small integer powers using FP16 comparison
                    else {
                        float bf = __half2float(b);
                        float abs_bf = fabsf(bf);

                        if (bf == floorf(bf) && abs_bf <= 10.0f) {
                            int n = (int)abs_bf;
                            half temp_result;

                            switch(n) {
                                case 0: temp_result = c_fp16_one; break;
                                case 1: temp_result = a; break;
                                case 2: temp_result = __hmul(a, a); break;
                                case 3: {
                                    half a2 = __hmul(a, a);
                                    temp_result = __hmul(a2, a);
                                    break;
                                }
                                case 4: {
                                    half a2 = __hmul(a, a);
                                    temp_result = __hmul(a2, a2);
                                    break;
                                }
                                case 5: {
                                    half a2 = __hmul(a, a);
                                    half a4 = __hmul(a2, a2);
                                    temp_result = __hmul(a4, a);
                                    break;
                                }
                                default: {
                                    temp_result = a;
                                    for (int i = 1; i < n; i++) {
                                        temp_result = __hmul(temp_result, a);
                                    }
                                }
                            }
                            result = (bf < 0.0f) ? __hdiv(c_fp16_one, temp_result) : temp_result;
                        } else {
                            float af = __half2float(a);
                            result = __float2half(powf(af, bf));
                        }
                    }
                    break;
                }
                default: result = __float2half(0.0f);
            }
            stack[sp++] = result;
        }
        else if (t == TokenType::Function) {
            if (sp < 1) return 0.0f;
            half a = stack[--sp];
            int op = op_codes[i];

            half result;
            switch (op) {
                case 10: {                                                         
                    #if __CUDA_ARCH__ >= 530
                    result = hsin(a);
                    #else
                    result = __float2half(sinf(__half2float(a)));
                    #endif
                    break;
                }
                case 11: {                                                        
                    #if __CUDA_ARCH__ >= 530
                    result = hcos(a);
                    #else
                    result = __float2half(cosf(__half2float(a)));
                    #endif
                    break;
                }
                case 12: {                                                        
                    #if __CUDA_ARCH__ >= 530
                    half eps = __float2half(1e-5f);
                    result = __hgt(a, eps) ? hlog10(a) : __float2half(-5.0f);
                    #else
                    float af = __half2float(a);
                    result = __float2half((af > 1e-5f) ? log10f(af) : -5.0f);
                    #endif
                    break;
                }
                case 13: {                                                       
                    #if __CUDA_ARCH__ >= 530
                    half eps = __float2half(1e-5f);
                    result = __hgt(a, eps) ? hlog(a) : __float2half(-11.5f);
                    #else
                    float af = __half2float(a);
                    result = __float2half((af > 1e-5f) ? logf(af) : -11.5f);
                    #endif
                    break;
                }
                case 14: {                                                       
                    #if __CUDA_ARCH__ >= 530
                    half max_val = __float2half(11.0f);
                    half clamped = __hlt(a, max_val) ? a : max_val;
                    result = hexp(clamped);
                    #else
                    float af = __half2float(a);
                    result = __float2half(expf(fminf(af, 11.0f)));
                    #endif
                    break;
                }
                case 15: {                                                       
                    #if __CUDA_ARCH__ >= 530
                    half zero = __float2half(0.0f);
                    result = __hgt(a, zero) ? hsqrt(a) : zero;
                    #else
                    float af = __half2float(a);
                    result = __float2half((af >= 0.0f) ? sqrtf(af) : 0.0f);
                    #endif
                    break;
                }
                case 16: {                                                        
                    result = __float2half(tanf(__half2float(a)));
                    break;
                }
                case 17: result = __habs(a); break;                             
                default: result = __float2half(0.0f);
            }
            stack[sp++] = result;
        }
    }

    return (sp > 0) ? __half2float(stack[0]) : 0.0f;
}

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
    float* __restrict__ d_block_sums,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    int global_block = blockIdx.x;
    int grid_blocks = gridDim.x;
    int blocks_per_term = (grid_blocks > num_terms) ? (grid_blocks / num_terms) : 1;
    int term_id = global_block / blocks_per_term;
    int block_in_term = global_block % blocks_per_term;
    if (term_id >= num_terms) return;

    const int expr_offset = term_id * expr_length;
    unsigned long long rng_state = seed + ((unsigned long long)term_id << 32) + tid + (unsigned long long)block_in_term;
    
    float volume = 1.0f;
    float bmin_f[10];
    float delta_f[10];
    #pragma unroll
    for (int d = 0; d < dimensions; ++d) {
        double lo = d_bounds_min[d];
        double hi = d_bounds_max[d];
        bmin_f[d] = (float)lo;
        delta_f[d] = (float)(hi - lo);
        volume *= delta_f[d];
    }
    
    float local_sum = 0.0f;
    
    const bool use_sobol = (d_sobol != nullptr);
    const size_t sobol_term_base = (size_t)term_id * samples_per_term * dimensions;
    size_t samples_per_block = (samples_per_term + blocks_per_term - 1) / blocks_per_term;
    size_t start = (size_t)block_in_term * samples_per_block;
    size_t end = start + samples_per_block;
    if (end > samples_per_term) end = samples_per_term;

    const float eps_f = 1e-12f;
    if (use_sobol) {
        for (size_t s = start + tid; s < end; s += block_size) {
            float vars[10];
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
        // Use vectorized RNG for reduced overhead
        if (dimensions == 4) {
            for (size_t s = start + tid; s < end; s += block_size) {
                float4 u4 = xorshift_random_float4(rng_state);
                float vars[10];
                vars[0] = bmin_f[0] + fminf(fmaxf(u4.x, eps_f), 1.0f - eps_f) * delta_f[0];
                vars[1] = bmin_f[1] + fminf(fmaxf(u4.y, eps_f), 1.0f - eps_f) * delta_f[1];
                vars[2] = bmin_f[2] + fminf(fmaxf(u4.z, eps_f), 1.0f - eps_f) * delta_f[2];
                vars[3] = bmin_f[3] + fminf(fmaxf(u4.w, eps_f), 1.0f - eps_f) * delta_f[3];
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
        } else if (dimensions == 2) {
            for (size_t s = start + tid; s < end; s += block_size) {
                float2 u2 = xorshift_random_float2(rng_state);
                float vars[10];
                vars[0] = bmin_f[0] + fminf(fmaxf(u2.x, eps_f), 1.0f - eps_f) * delta_f[0];
                vars[1] = bmin_f[1] + fminf(fmaxf(u2.y, eps_f), 1.0f - eps_f) * delta_f[1];
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
            // Fallback for other dimensions
            for (size_t s = start + tid; s < end; s += block_size) {
                float vars[10];
                #pragma unroll
                for (int d = 0; d < dimensions; ++d) {
                    float u = xorshift_random_float(rng_state);
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
    }

    // Block-level reduction using CUB
    using BlockReduceF = cub::BlockReduce<float, 512>;
    __shared__ typename BlockReduceF::TempStorage temp_storage_f;
    float block_sum = BlockReduceF(temp_storage_f).Sum(local_sum);
    if (threadIdx.x == 0) {
        int blocks_per_term = (grid_blocks > num_terms) ? (grid_blocks / num_terms) : 1;
        int out_idx = term_id * blocks_per_term + block_in_term;
        d_block_sums[out_idx] = block_sum;
    }
}

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
    double* __restrict__ d_block_sums,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    int global_block = blockIdx.x;
    int term_id = global_block / 1; // default 1 block per term unless host requests more
    int block_in_term = 0;
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

    double local_sum = 0.0;

    const bool use_sobol = (d_sobol != nullptr);
    const size_t sobol_term_base = (size_t)term_id * samples_per_term * dimensions;

    // compute per-block sample range
    int blocks_per_term = (gridDim.x > num_terms) ? (gridDim.x / num_terms) : 1;
    size_t samples_per_block = (samples_per_term + blocks_per_term - 1) / blocks_per_term;
    size_t start = (size_t)block_in_term * samples_per_block;
    size_t end = start + samples_per_block;
    if (end > samples_per_term) end = samples_per_term;

    const double eps_d = 1e-15;
    if (use_sobol) {
        for (size_t s = start + tid; s < end; s += block_size) {
            double vars[10];
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
        // Use vectorized RNG for reduced overhead
        if (dimensions == 4) {
            for (size_t s = start + tid; s < end; s += block_size) {
                double4 u4 = xorshift_random_double4(rng_state);
                double vars[10];
                vars[0] = d_bounds_min[0] + fmin(fmax(u4.x, eps_d), 1.0 - eps_d) * (d_bounds_max[0] - d_bounds_min[0]);
                vars[1] = d_bounds_min[1] + fmin(fmax(u4.y, eps_d), 1.0 - eps_d) * (d_bounds_max[1] - d_bounds_min[1]);
                vars[2] = d_bounds_min[2] + fmin(fmax(u4.z, eps_d), 1.0 - eps_d) * (d_bounds_max[2] - d_bounds_min[2]);
                vars[3] = d_bounds_min[3] + fmin(fmax(u4.w, eps_d), 1.0 - eps_d) * (d_bounds_max[3] - d_bounds_min[3]);
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
        } else if (dimensions == 2) {
            for (size_t s = start + tid; s < end; s += block_size) {
                double2 u2 = xorshift_random_double2(rng_state);
                double vars[10];
                vars[0] = d_bounds_min[0] + fmin(fmax(u2.x, eps_d), 1.0 - eps_d) * (d_bounds_max[0] - d_bounds_min[0]);
                vars[1] = d_bounds_min[1] + fmin(fmax(u2.y, eps_d), 1.0 - eps_d) * (d_bounds_max[1] - d_bounds_min[1]);
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
                double vars[10];
                #pragma unroll
                for (int d = 0; d < dimensions; ++d) {
                    double u = xorshift_random_double(rng_state);
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
    }

    // Block-level reduction using CUB for double
    using BlockReduceD = cub::BlockReduce<double, 512>;
    __shared__ typename BlockReduceD::TempStorage temp_storage_d;
    double block_sum = BlockReduceD(temp_storage_d).Sum(local_sum);
    if (threadIdx.x == 0) {
        int blocks_per_term = (gridDim.x > num_terms) ? (gridDim.x / num_terms) : 1;
        int out_idx = term_id * blocks_per_term + block_in_term;
        d_block_sums[out_idx] = block_sum;
    }
}

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
    float* __restrict__ d_block_sums,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    int global_block = blockIdx.x;
    int grid_blocks = gridDim.x;
    int blocks_per_term = (grid_blocks > num_terms) ? (grid_blocks / num_terms) : 1;
    int term_id = global_block / blocks_per_term;
    int block_in_term = global_block % blocks_per_term;
    if (term_id >= num_terms) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int expr_offset = term_id * expr_length;

    unsigned long long rng_state = seed + ((unsigned long long)term_id << 32) + tid + (unsigned long long)block_in_term;
    
    float volume = 1.0f;
    float bmin_f[10];
    float delta_f[10];
    #pragma unroll
    for (int d = 0; d < dimensions; ++d) {
        bmin_f[d] = (float)d_bounds_min[d];
        delta_f[d] = (float)(d_bounds_max[d] - d_bounds_min[d]);
        volume *= delta_f[d];
    }

    float local_sum = 0.0f;

    const bool use_sobol = (d_sobol != nullptr);
    size_t samples_per_block = (samples_per_term + blocks_per_term - 1) / blocks_per_term;
    size_t start = (size_t)block_in_term * samples_per_block;
    size_t end = start + samples_per_block;
    if (end > samples_per_term) end = samples_per_term;
    const size_t sobol_term_base = (size_t)term_id * samples_per_term * dimensions;
    const float eps_f = 1e-12f;

    if (use_sobol) {
        for (size_t s = start + tid; s < end; s += block_size) {
            float vars[10];
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
        // Use vectorized RNG for reduced overhead
        if (dimensions == 4) {
            for (size_t s = start + tid; s < end; s += block_size) {
                float4 u4 = xorshift_random_float4(rng_state);
                float vars[10];
                vars[0] = bmin_f[0] + fminf(fmaxf(u4.x, eps_f), 1.0f - eps_f) * delta_f[0];
                vars[1] = bmin_f[1] + fminf(fmaxf(u4.y, eps_f), 1.0f - eps_f) * delta_f[1];
                vars[2] = bmin_f[2] + fminf(fmaxf(u4.z, eps_f), 1.0f - eps_f) * delta_f[2];
                vars[3] = bmin_f[3] + fminf(fmaxf(u4.w, eps_f), 1.0f - eps_f) * delta_f[3];
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
        } else if (dimensions == 2) {
            for (size_t s = start + tid; s < end; s += block_size) {
                float2 u2 = xorshift_random_float2(rng_state);
                float vars[10];
                vars[0] = bmin_f[0] + fminf(fmaxf(u2.x, eps_f), 1.0f - eps_f) * delta_f[0];
                vars[1] = bmin_f[1] + fminf(fmaxf(u2.y, eps_f), 1.0f - eps_f) * delta_f[1];
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
            for (size_t s = start + tid; s < end; s += block_size) {
                float vars[10];
                #pragma unroll
                for (int d = 0; d < dimensions; ++d) {
                    float u = xorshift_random_float(rng_state);
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
    }

    using BlockReduceF = cub::BlockReduce<float, 512>;
    __shared__ typename BlockReduceF::TempStorage temp_storage_f;
    float block_sum = BlockReduceF(temp_storage_f).Sum(local_sum);
    if (threadIdx.x == 0) {
        int out_idx = term_id * blocks_per_term + block_in_term;
        d_block_sums[out_idx] = block_sum;
    }
}

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
    double* __restrict__ d_block_sums,
    const int* __restrict__ d_precisions,
    unsigned long long seed,
    const double* __restrict__ d_sobol)
{
    int global_block = blockIdx.x;
    int grid_blocks = gridDim.x;
    int blocks_per_term = (grid_blocks > num_terms) ? (grid_blocks / num_terms) : 1;
    int term_id = global_block / blocks_per_term;
    int block_in_term = global_block % blocks_per_term;
    if (term_id >= num_terms) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const int prec = d_precisions[term_id];
    const int expr_offset = term_id * expr_length;
    const size_t samples_this_term = d_samples_per_term[term_id];

    const double* bounds_min = d_bounds_min_per_term + (size_t)term_id * dimensions;
    const double* bounds_max = d_bounds_max_per_term + (size_t)term_id * dimensions;

    unsigned long long rng_state = seed + ((unsigned long long)term_id << 32) + tid + (unsigned long long)block_in_term;

    const bool use_sobol = (d_sobol != nullptr);
    const size_t sobol_offset = use_sobol ? d_sample_offsets[term_id] : 0;

    if (prec == 2) {
        double volume = 1.0;
        for (int d = 0; d < dimensions; ++d) {
            volume *= (bounds_max[d] - bounds_min[d]);
        }

        double local_sum = 0.0;

        for (size_t s = tid; s < samples_this_term; s += block_size) {
            double vars[10];

            for (int d = 0; d < dimensions; ++d) {
                double u = use_sobol ? d_sobol[(sobol_offset + s) * dimensions + d]
                                     : xorshift_random_double(rng_state);
                vars[d] = bounds_min[d] + u * (bounds_max[d] - bounds_min[d]);
            }

            local_sum += eval_expr_device_fp64(
                d_types + expr_offset, d_constants + expr_offset,
                d_var_indices + expr_offset, d_op_codes + expr_offset,
                expr_length, vars);
        }

        using BlockReduceD = cub::BlockReduce<double, 512>;
        __shared__ typename BlockReduceD::TempStorage temp_storage_d;
        double block_sum = BlockReduceD(temp_storage_d).Sum(local_sum);
        if (threadIdx.x == 0) {
            int out_idx = term_id * blocks_per_term + block_in_term;
            d_block_sums[out_idx] = block_sum;
        }

    } else {
        float volume = 1.0f;
        for (int d = 0; d < dimensions; ++d) {
            volume *= (float)(bounds_max[d] - bounds_min[d]);
        }

        float local_sum = 0.0f;

        for (size_t s = tid; s < samples_this_term; s += block_size) {
            float vars[10];

            for (int d = 0; d < dimensions; ++d) {
                float u = use_sobol ? (float)d_sobol[(sobol_offset + s) * dimensions + d]
                                    : xorshift_random_float(rng_state);
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

        using BlockReduceF = cub::BlockReduce<float, 512>;
        __shared__ typename BlockReduceF::TempStorage temp_storage_f;
        float block_sum = BlockReduceF(temp_storage_f).Sum(local_sum);
        if (threadIdx.x == 0) {
            int out_idx = term_id * blocks_per_term + block_in_term;
            d_block_sums[out_idx] = (double)block_sum;
        }
    }
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

    // Use Xorshift for fastest random generation
    double* d_sobol = nullptr;

    int optimal_threads = config.threads_per_block; int blocks_per_term = 1;
    if (samples > 10'000'000) blocks_per_term = 4;
    dim3 blocks(num_terms * blocks_per_term); dim3 threads(optimal_threads);

    static unsigned long long base_seed = 0x123456789abcdefULL;
    base_seed += 0x9e3779b97f4a7c15ULL;

    std::vector<T> host_results(num_terms);

    // Use runtime check instead of constexpr if for C++14 compatibility
    if (std::is_same<T, double>::value) {
        double* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, num_terms * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_results, 0, num_terms * sizeof(double)));

        // allocate per-block partial sums (one entry per launched block)
        int blocks_per_term = (blocks.x > (unsigned)num_terms) ? (blocks.x / num_terms) : 1;
        size_t block_sums_elems = (size_t)num_terms * blocks_per_term;
        double* d_block_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_elems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_block_sums, 0, block_sums_elems * sizeof(double)));

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
            d_results, d_block_sums, base_seed, d_sobol);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaDeviceSynchronize());
        // copy per-block sums and reduce on host to produce final per-term results
        std::vector<double> h_block_sums(block_sums_elems);
        CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                              block_sums_elems * sizeof(double), cudaMemcpyDeviceToHost));

        // compute per-term volume
        double volume = 1.0;
        for (int d = 0; d < dimensions; ++d) volume *= (bounds_max[d] - bounds_min[d]);

        for (int i = 0; i < num_terms; ++i) {
            double s = 0.0;
            for (int b = 0; b < blocks_per_term; ++b) s += h_block_sums[(size_t)i * blocks_per_term + b];
            host_results[i] = (volume / (double)samples) * s;
        }

        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_constants_d));

    } else if (std::is_same<T, float>::value) {
        float* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, num_terms * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_results, 0, num_terms * sizeof(float)));

        // allocate per-block partial sums
        int blocks_per_term = (blocks.x > (unsigned)num_terms) ? (blocks.x / num_terms) : 1;
        size_t block_sums_elems = (size_t)num_terms * blocks_per_term;
        float* d_block_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_block_sums, 0, block_sums_elems * sizeof(float)));

        mc_integrate_kernel_fp32_optimized<<<blocks, threads>>>(
            d_types, d_constants, d_var_indices, d_op_codes,
            expr_length, d_bounds_min, d_bounds_max,
            dimensions, samples, num_terms,
            d_results, d_block_sums, base_seed, d_sobol);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_block_sums(block_sums_elems);
        CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                              block_sums_elems * sizeof(float), cudaMemcpyDeviceToHost));

        float volume = 1.0f;
        for (int d = 0; d < dimensions; ++d) volume *= (float)(bounds_max[d] - bounds_min[d]);

        for (int i = 0; i < num_terms; ++i) {
            double s = 0.0;
            for (int b = 0; b < blocks_per_term; ++b) s += h_block_sums[(size_t)i * blocks_per_term + b];
            host_results[i] = static_cast<T>((volume / (float)samples) * (float)s);
        }

        CUDA_CHECK(cudaFree(d_block_sums));
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

    int optimal_threads = config.threads_per_block;
    int blocks_per_term = 1;
    if (samples > 10'000'000) blocks_per_term = 4;
    dim3 blocks(num_terms * blocks_per_term);
    dim3 threads(optimal_threads);

    // per-block partial sums (one entry per launched block)
    size_t block_sums_elems = (size_t)num_terms * blocks_per_term;
    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, block_sums_elems * sizeof(float)));

    // Use Xorshift for fastest random generation
    double* d_sobol = nullptr;

    static unsigned long long seed = 0xdeadbeefcafebabeULL;
    seed += 0x9e3779b97f4a7c15ULL;

    mc_integrate_kernel_fp16_optimized<<<blocks, threads>>> (
        d_types, d_constants, d_var_indices, d_op_codes,
        expr_length, d_bounds_min, d_bounds_max,
        dimensions, samples, num_terms,
        d_results, d_block_sums, seed, d_sobol);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> results(num_terms);
    // copy per-block sums and reduce on host
    std::vector<float> h_block_sums(block_sums_elems);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          block_sums_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float volume = 1.0f;
    for (int d = 0; d < dimensions; ++d) volume *= (float)(bounds_max[d] - bounds_min[d]);
    for (int i = 0; i < num_terms; ++i) {
        float s = 0.0f;
        for (int b = 0; b < blocks_per_term; ++b) s += h_block_sums[(size_t)i * blocks_per_term + b];
        results[i] = (volume / (float)samples) * s;
    }

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

    // Use Xorshift for fastest random generation
    double* d_sobol = nullptr;

    double* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_terms * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_results, 0, num_terms * sizeof(double)));

    dim3 blocks(num_terms);
    dim3 threads(config.threads_per_block);

    static unsigned long long seed = 0xdeadbeef12345678ULL;
    seed += 0x9e3779b97f4a7c15ULL;
    int blocks_per_term = (blocks.x > (unsigned)num_terms) ? (blocks.x / num_terms) : 1;
    size_t block_sums_elems = (size_t)num_terms * blocks_per_term;
    double* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, block_sums_elems * sizeof(double)));

    mc_integrate_kernel_mixed_optimized<<<blocks, threads>>>(
        d_types, d_constants, d_var_indices, d_op_codes, expr_length,
        d_bounds_min, d_bounds_max, dimensions,
        d_samples, d_offsets, total_samples, num_terms,
        d_results, d_block_sums, d_precisions, seed, d_sobol);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> results(num_terms);
    std::vector<double> h_block_sums(block_sums_elems);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          block_sums_elems * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_terms; ++i) {
        double volume = 1.0;
        for (int d = 0; d < dimensions; ++d) {
            double lo = flat_min[i * dimensions + d];
            double hi = flat_max[i * dimensions + d];
            volume *= (hi - lo);
        }
        double s = 0.0;
        for (int b = 0; b < blocks_per_term; ++b) s += h_block_sums[(size_t)i * blocks_per_term + b];
        int samples_this_term = samp_i[i];
        results[i] = (samples_this_term > 0) ? (volume / (double)samples_this_term) * s : 0.0;
    }

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

// Simplified API wrapper that matches header declaration
std::vector<double> monte_carlo_integrate_nd_cuda_mixed(
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& exprs,
    const std::vector<Precision>& precisions,
    const std::vector<size_t>& samples_per_term,
    const GPUConfig& config)
{
    // Call the existing implementation with empty per-term bounds
    // (it will use global bounds for all terms)
    std::vector<std::vector<double>> empty_bounds_min;
    std::vector<std::vector<double>> empty_bounds_max;

    return monte_carlo_integrate_nd_cuda_batch_mixed(
        0,  // samples parameter (unused when samples_per_term is provided)
        bounds_min,
        bounds_max,
        empty_bounds_min,
        empty_bounds_max,
        samples_per_term,
        exprs,
        precisions,
        config
    );
}

template std::vector<float> monte_carlo_integrate_nd_cuda_batch<float>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);

template std::vector<double> monte_carlo_integrate_nd_cuda_batch<double>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);