#include "cuda_integration.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cmath>

// ============================================================================
// CONSTANT MEMORY
// ============================================================================
__constant__ half c_fp16_zero;
__constant__ half c_fp16_one;
__constant__ half c_fp16_half;
__constant__ half c_fp16_eps;

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
    config.threads_per_block = 256;
    config.blocks_per_sm = 8;
    return config;
}

// ============================================================================
// SOBOL QMC STATE MANAGEMENT
// ============================================================================
struct SobolQMC {
    curandStateSobol32_t* states;
    curandDirectionVectors32_t* directions;
    int num_states;
    int dimensions;
    bool initialized;

    SobolQMC() : states(nullptr), directions(nullptr), num_states(0),
                 dimensions(0), initialized(false) {}

    void init(int n_states, int dims) {
        if (initialized) return;

        num_states = n_states;
        dimensions = dims;

        // Get direction vectors from cuRAND
        curandDirectionVectors32_t* h_host_vectors;
        curandStatus_t curand_status = curandGetDirectionVectors32(
            &h_host_vectors,
            CURAND_DIRECTION_VECTORS_32_JOEKUO6);
        if (curand_status != CURAND_STATUS_SUCCESS) abort();

        // Copy first 'dims' direction vectors to device
        CUDA_CHECK(cudaMalloc(&directions, dims * sizeof(curandDirectionVectors32_t)));
        CUDA_CHECK(cudaMemcpy(directions, h_host_vectors,
                             dims * sizeof(curandDirectionVectors32_t),
                             cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&states, n_states * sizeof(curandStateSobol32_t)));
        initialized = true;
    }

    void cleanup() {
        if (states) cudaFree(states);
        if (directions) cudaFree(directions);
        states = nullptr;
        directions = nullptr;
        initialized = false;
    }

    ~SobolQMC() { cleanup(); }
};

// Global Sobol QMC state (initialized once per program)
static SobolQMC g_sobol_qmc;

// Initialize Sobol states kernel
__global__ void init_sobol_kernel(
    curandStateSobol32_t* states,
    curandDirectionVectors32_t* directions,
    int dims,
    unsigned long long offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = tid % dims;
    curand_init(directions[dim_idx], offset + tid, &states[tid]);
}

// ============================================================================
// DEVICE EVALUATION FUNCTIONS (UNIFIED)
// ============================================================================
template<typename T>
__device__ __forceinline__ T eval_expr_device(
    const TokenType* types,
    const float* constants,
    const int* var_indices,
    const int* op_codes,
    int length,
    const T* vars)
{
    T stack[32];
    int sp = 0;

    for (int i = 0; i < length; ++i) {
        TokenType t = types[i];

        if (t == TokenType::Number) {
            stack[sp++] = (T)constants[i];
        }
        else if (t == TokenType::Variable) {
            int idx = var_indices[i];
            stack[sp++] = (idx >= 0 && idx < 10) ? vars[idx] : (T)0;
        }
        else if (t == TokenType::Operator) {
            if (sp < 2) return (T)0;
            T b = stack[--sp];
            T a = stack[--sp];
            int op = op_codes[i];

            T result;
            if constexpr (std::is_same<T, double>::value) {
                switch (op) {
                    case 0: result = a + b; break;
                    case 1: result = a - b; break;
                    case 2: result = a * b; break;
                    case 3: result = (fabs(b) > 1e-15) ? (a / b) : 0.0; break;
                    case 4: result = pow(a, b); break;
                    default: result = 0.0;
                }
            } else {
                switch (op) {
                    case 0: result = a + b; break;
                    case 1: result = a - b; break;
                    case 2: result = a * b; break;
                    case 3: result = (fabsf(b) > 1e-10f) ? (a / b) : 0.0f; break;
                    case 4: result = powf(a, b); break;
                    default: result = 0.0f;
                }
            }
            stack[sp++] = result;
        }
        else if (t == TokenType::Function) {
            if (sp < 1) return (T)0;
            T a = stack[--sp];
            int op = op_codes[i];

            T result;
            if constexpr (std::is_same<T, double>::value) {
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
            } else {
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
            }
            stack[sp++] = result;
        }
    }

    return (sp > 0) ? stack[0] : (T)0;
}

// FP16 specialization
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
            stack[sp++] = (idx >= 0 && idx < 10) ? __float2half(vars[idx]) : c_fp16_zero;
        }
        else if (t == TokenType::Operator) {
            if (sp < 2) return 0.0f;
            half b = stack[--sp];
            half a = stack[--sp];
            int op = op_codes[i];

            half result;
            switch (op) {
                case 0: result = __hadd(a, b); break;
                case 1: result = __hsub(a, b); break;
                case 2: result = __hmul(a, b); break;
                case 3: result = __hgt(__habs(b), c_fp16_eps) ? __hdiv(a, b) : c_fp16_zero; break;
                case 4: {
                    // Power optimization
                    if (__heq(b, c_fp16_half)) {
                        #if __CUDA_ARCH__ >= 530
                        result = __hgt(a, c_fp16_zero) ? hsqrt(a) : c_fp16_zero;
                        #else
                        float af = __half2float(a);
                        result = __float2half((af >= 0.0f) ? sqrtf(af) : 0.0f);
                        #endif
                    } else {
                        result = __float2half(powf(__half2float(a), __half2float(b)));
                    }
                    break;
                }
                default: result = c_fp16_zero;
            }
            stack[sp++] = result;
        }
        else if (t == TokenType::Function) {
            if (sp < 1) return 0.0f;
            half a = stack[--sp];
            int op = op_codes[i];

            half result;
            switch (op) {
                #if __CUDA_ARCH__ >= 530
                case 10: result = hsin(a); break;
                case 11: result = hcos(a); break;
                case 12: result = __hgt(a, c_fp16_eps) ? hlog10(a) : __float2half(-5.0f); break;
                case 13: result = __hgt(a, c_fp16_eps) ? hlog(a) : __float2half(-11.5f); break;
                case 14: result = hexp(__hlt(a, __float2half(11.0f)) ? a : __float2half(11.0f)); break;
                case 15: result = __hgt(a, c_fp16_zero) ? hsqrt(a) : c_fp16_zero; break;
                #else
                case 10: result = __float2half(sinf(__half2float(a))); break;
                case 11: result = __float2half(cosf(__half2float(a))); break;
                case 12: result = __float2half(log10f(__half2float(a))); break;
                case 13: result = __float2half(logf(__half2float(a))); break;
                case 14: result = __float2half(expf(__half2float(a))); break;
                case 15: result = __float2half(sqrtf(__half2float(a))); break;
                #endif
                case 16: result = __float2half(tanf(__half2float(a))); break;
                case 17: result = __habs(a); break;
                default: result = c_fp16_zero;
            }
            stack[sp++] = result;
        }
    }

    return (sp > 0) ? __half2float(stack[0]) : 0.0f;
}

// ============================================================================
// UNIFIED MONTE CARLO KERNEL WITH SOBOL QMC
// ============================================================================
template<typename T, bool USE_FP16 = false>
__global__ void mc_integrate_kernel(
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
    T* __restrict__ d_block_sums,
    curandStateSobol32_t* __restrict__ d_sobol_states)
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

    // Cache expression data in shared memory for faster access
    extern __shared__ char shared_mem[];
    TokenType* s_types = (TokenType*)shared_mem;
    float* s_constants = (float*)(s_types + expr_length);
    int* s_var_indices = (int*)(s_constants + expr_length);
    int* s_op_codes = (int*)(s_var_indices + expr_length);

    // Cooperative loading of expression data
    for (int i = tid; i < expr_length; i += block_size) {
        s_types[i] = d_types[expr_offset + i];
        s_constants[i] = d_constants[expr_offset + i];
        s_var_indices[i] = d_var_indices[expr_offset + i];
        s_op_codes[i] = d_op_codes[expr_offset + i];
    }
    __syncthreads();

    // Precompute bounds in registers
    T volume = (T)1.0;
    T bmin[10], delta[10];
    #pragma unroll
    for (int d = 0; d < dimensions && d < 10; ++d) {
        double lo = d_bounds_min[d];
        double hi = d_bounds_max[d];
        bmin[d] = (T)lo;
        delta[d] = (T)(hi - lo);
        volume *= delta[d];
    }

    T local_sum = (T)0.0;

    // Compute per-block sample range
    size_t samples_per_block = (samples_per_term + blocks_per_term - 1) / blocks_per_term;
    size_t start = (size_t)block_in_term * samples_per_block;
    size_t end = start + samples_per_block;
    if (end > samples_per_term) end = samples_per_term;

    const T eps = std::is_same<T, double>::value ? (T)1e-15 : (T)1e-12f;

    // Sobol QMC sampling with loop unrolling and register optimization
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateSobol32_t local_state = d_sobol_states[state_idx];

    // Process multiple samples per iteration to amortize memory latency
    constexpr int SAMPLES_PER_ITER = 4;
    size_t s = start + tid;

    // ANTITHETIC VARIATES: For each Sobol sample, also compute (1-u)
    // This doubles compute per memory access, helping with memory-bound performance
    // Also improves convergence by reducing variance
    for (; s + SAMPLES_PER_ITER * block_size <= end; s += SAMPLES_PER_ITER * block_size) {
        #pragma unroll
        for (int iter = 0; iter < SAMPLES_PER_ITER; ++iter) {
            T vars[10];
            T vars_anti[10];  // Antithetic sample

            // Generate Sobol samples and antithetic pairs
            #pragma unroll
            for (int d = 0; d < dimensions && d < 10; ++d) {
                float u = curand_uniform(&local_state);
                u = fminf(fmaxf(u, 1e-12f), 1.0f - 1e-12f);
                vars[d] = bmin[d] + (T)u * delta[d];
                // Antithetic: use (1-u) instead of u
                vars_anti[d] = bmin[d] + (T)(1.0f - u) * delta[d];
            }

            // Evaluate function using cached expression data
            T f_val, f_val_anti;
            if constexpr (USE_FP16) {
                float vars_f[10], vars_anti_f[10];
                for (int d = 0; d < dimensions; ++d) {
                    vars_f[d] = (float)vars[d];
                    vars_anti_f[d] = (float)vars_anti[d];
                }
                f_val = (T)eval_expr_device_fp16(
                    s_types, s_constants, s_var_indices, s_op_codes,
                    expr_length, vars_f
                );
                f_val_anti = (T)eval_expr_device_fp16(
                    s_types, s_constants, s_var_indices, s_op_codes,
                    expr_length, vars_anti_f
                );
            } else {
                f_val = eval_expr_device<T>(
                    s_types, s_constants, s_var_indices, s_op_codes,
                    expr_length, vars
                );
                f_val_anti = eval_expr_device<T>(
                    s_types, s_constants, s_var_indices, s_op_codes,
                    expr_length, vars_anti
                );
            }

            // Average the antithetic pairs (reduces variance)
            local_sum += (f_val + f_val_anti) * (T)0.5;
        }
    }

    // Handle remaining samples with antithetic variates
    for (; s < end; s += block_size) {
        T vars[10];
        T vars_anti[10];

        // Generate Sobol samples and antithetic pairs
        #pragma unroll
        for (int d = 0; d < dimensions && d < 10; ++d) {
            float u = curand_uniform(&local_state);
            u = fminf(fmaxf(u, 1e-12f), 1.0f - 1e-12f);
            vars[d] = bmin[d] + (T)u * delta[d];
            vars_anti[d] = bmin[d] + (T)(1.0f - u) * delta[d];
        }

        // Evaluate function using cached expression data
        T f_val, f_val_anti;
        if constexpr (USE_FP16) {
            float vars_f[10], vars_anti_f[10];
            for (int d = 0; d < dimensions; ++d) {
                vars_f[d] = (float)vars[d];
                vars_anti_f[d] = (float)vars_anti[d];
            }
            f_val = (T)eval_expr_device_fp16(
                s_types, s_constants, s_var_indices, s_op_codes,
                expr_length, vars_f
            );
            f_val_anti = (T)eval_expr_device_fp16(
                s_types, s_constants, s_var_indices, s_op_codes,
                expr_length, vars_anti_f
            );
        } else {
            f_val = eval_expr_device<T>(
                s_types, s_constants, s_var_indices, s_op_codes,
                expr_length, vars
            );
            f_val_anti = eval_expr_device<T>(
                s_types, s_constants, s_var_indices, s_op_codes,
                expr_length, vars_anti
            );
        }

        local_sum += (f_val + f_val_anti) * (T)0.5;
    }

    // Save state back
    d_sobol_states[state_idx] = local_state;

    // Block-level reduction
    using BlockReduce = cub::BlockReduce<T, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T block_sum = BlockReduce(temp_storage).Sum(local_sum);

    if (threadIdx.x == 0) {
        int out_idx = term_id * blocks_per_term + block_in_term;
        d_block_sums[out_idx] = block_sum;
    }
}

// ============================================================================
// MIXED PRECISION KERNEL
// ============================================================================
__global__ void mc_integrate_kernel_mixed(
    const TokenType* __restrict__ d_types,
    const float* __restrict__ d_constants,
    const int* __restrict__ d_var_indices,
    const int* __restrict__ d_op_codes,
    int expr_length,
    const double* __restrict__ d_bounds_min_per_term,
    const double* __restrict__ d_bounds_max_per_term,
    int dimensions,
    const int* __restrict__ d_samples_per_term,
    int num_terms,
    int blocks_per_term,
    double* __restrict__ d_block_sums,
    const int* __restrict__ d_precisions,
    curandStateSobol32_t* __restrict__ d_sobol_states)
{
    int global_block = blockIdx.x;
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

    size_t samples_per_block = (samples_this_term + blocks_per_term - 1) / blocks_per_term;
    size_t start = (size_t)block_in_term * samples_per_block;
    size_t end = start + samples_per_block;
    if (end > samples_this_term) end = samples_this_term;

    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateSobol32_t local_state = d_sobol_states[state_idx];

    double volume = 1.0;
    for (int d = 0; d < dimensions; ++d) {
        volume *= (bounds_max[d] - bounds_min[d]);
    }

    if (prec == 2) {
        double local_sum = 0.0;
        for (size_t s = start + tid; s < end; s += block_size) {
            double vars[10];
            for (int d = 0; d < dimensions; ++d) {
                float u = curand_uniform(&local_state);
                vars[d] = bounds_min[d] + u * (bounds_max[d] - bounds_min[d]);
            }
            local_sum += eval_expr_device<double>(
                d_types + expr_offset, d_constants + expr_offset,
                d_var_indices + expr_offset, d_op_codes + expr_offset,
                expr_length, vars);
        }

        using BlockReduceD = cub::BlockReduce<double, 256>;
        __shared__ typename BlockReduceD::TempStorage temp_storage_d;
        double block_sum = BlockReduceD(temp_storage_d).Sum(local_sum);
        if (threadIdx.x == 0) {
            int out_idx = term_id * blocks_per_term + block_in_term;
            d_block_sums[out_idx] = block_sum;
        }
    } else {
        float local_sum = 0.0f;
        for (size_t s = start + tid; s < end; s += block_size) {
            float vars[10];
            for (int d = 0; d < dimensions; ++d) {
                float u = curand_uniform(&local_state);
                vars[d] = (float)bounds_min[d] + u * (float)(bounds_max[d] - bounds_min[d]);
            }

            if (prec == 1) {
                local_sum += eval_expr_device<float>(
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

        using BlockReduceF = cub::BlockReduce<float, 256>;
        __shared__ typename BlockReduceF::TempStorage temp_storage_f;
        float block_sum = BlockReduceF(temp_storage_f).Sum(local_sum);
        if (threadIdx.x == 0) {
            int out_idx = term_id * blocks_per_term + block_in_term;
            d_block_sums[out_idx] = (double)block_sum;
        }
    }

    d_sobol_states[state_idx] = local_state;
}

// ============================================================================
// HOST INTEGRATION FUNCTIONS
// ============================================================================
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

    int blocks_per_term = config.sm_count * config.blocks_per_sm / num_terms;
    blocks_per_term = std::min(std::max(blocks_per_term, 4), 32);
    int total_blocks = num_terms * blocks_per_term;
    int needed_states = total_blocks * config.threads_per_block;

    if (!g_sobol_qmc.initialized || g_sobol_qmc.num_states < needed_states) {
        if (g_sobol_qmc.initialized) g_sobol_qmc.cleanup();

        g_sobol_qmc.init(needed_states, dimensions);

        // Initialize Sobol states
        dim3 init_blocks((needed_states + 255) / 256);
        dim3 init_threads(256);
        init_sobol_kernel<<<init_blocks, init_threads>>>(
            g_sobol_qmc.states, g_sobol_qmc.directions, dimensions, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Allocate and copy bounds
    double *d_bounds_min, *d_bounds_max;
    CUDA_CHECK(cudaMalloc(&d_bounds_min, dimensions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bounds_max, dimensions * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_bounds_min, bounds_min.data(),
                          dimensions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bounds_max, bounds_max.data(),
                          dimensions * sizeof(double), cudaMemcpyHostToDevice));

    // Prepare expressions
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

    dim3 blocks(num_terms * blocks_per_term);
    dim3 threads(config.threads_per_block);

    size_t block_sums_elems = (size_t)num_terms * blocks_per_term;
    T* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_elems * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, block_sums_elems * sizeof(T)));

    // Calculate shared memory size for expression caching
    size_t shared_mem_size = expr_length * (sizeof(TokenType) + sizeof(float) + 2 * sizeof(int));

    // Launch kernel with shared memory
    mc_integrate_kernel<T, false><<<blocks, threads, shared_mem_size>>>(
        d_types, d_constants, d_var_indices, d_op_codes,
        expr_length, d_bounds_min, d_bounds_max,
        dimensions, samples, num_terms,
        d_block_sums, g_sobol_qmc.states);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduce results
    std::vector<T> h_block_sums(block_sums_elems);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          block_sums_elems * sizeof(T), cudaMemcpyDeviceToHost));

    double volume = 1.0;
    for (int d = 0; d < dimensions; ++d)
        volume *= (bounds_max[d] - bounds_min[d]);

    std::vector<T> host_results(num_terms);
    for (int i = 0; i < num_terms; ++i) {
        double s = 0.0;
        for (int b = 0; b < blocks_per_term; ++b)
            s += (double)h_block_sums[(size_t)i * blocks_per_term + b];
        host_results[i] = (T)((volume / (double)samples) * s);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_types));
    CUDA_CHECK(cudaFree(d_constants));
    CUDA_CHECK(cudaFree(d_var_indices));
    CUDA_CHECK(cudaFree(d_op_codes));
    CUDA_CHECK(cudaFree(d_bounds_min));
    CUDA_CHECK(cudaFree(d_bounds_max));

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

    int blocks_per_term = config.sm_count * config.blocks_per_sm / num_terms;
    blocks_per_term = std::min(std::max(blocks_per_term, 4), 32);
    int total_blocks = num_terms * blocks_per_term;
    int needed_states = total_blocks * config.threads_per_block;

    if (!g_sobol_qmc.initialized || g_sobol_qmc.num_states < needed_states) {
        if (g_sobol_qmc.initialized) g_sobol_qmc.cleanup();

        g_sobol_qmc.init(needed_states, dimensions);

        dim3 init_blocks((needed_states + 255) / 256);
        dim3 init_threads(256);
        init_sobol_kernel<<<init_blocks, init_threads>>>(
            g_sobol_qmc.states, g_sobol_qmc.directions, dimensions, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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

    dim3 blocks(num_terms * blocks_per_term);
    dim3 threads(config.threads_per_block);

    size_t block_sums_elems = (size_t)num_terms * blocks_per_term;
    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, block_sums_elems * sizeof(float)));

    // Calculate shared memory size for expression caching
    size_t shared_mem_size = expr_length * (sizeof(TokenType) + sizeof(float) + 2 * sizeof(int));

    mc_integrate_kernel<float, true><<<blocks, threads, shared_mem_size>>>(
        d_types, d_constants, d_var_indices, d_op_codes,
        expr_length, d_bounds_min, d_bounds_max,
        dimensions, samples, num_terms,
        d_block_sums, g_sobol_qmc.states);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_block_sums(block_sums_elems);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          block_sums_elems * sizeof(float), cudaMemcpyDeviceToHost));

    float volume = 1.0f;
    for (int d = 0; d < dimensions; ++d)
        volume *= (float)(bounds_max[d] - bounds_min[d]);

    std::vector<float> results(num_terms);
    for (int i = 0; i < num_terms; ++i) {
        float s = 0.0f;
        for (int b = 0; b < blocks_per_term; ++b)
            s += h_block_sums[(size_t)i * blocks_per_term + b];
        results[i] = (volume / (float)samples) * s;
    }

    CUDA_CHECK(cudaFree(d_types));
    CUDA_CHECK(cudaFree(d_constants));
    CUDA_CHECK(cudaFree(d_var_indices));
    CUDA_CHECK(cudaFree(d_op_codes));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_bounds_min));
    CUDA_CHECK(cudaFree(d_bounds_max));

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

    int blocks_per_term = config.sm_count * config.blocks_per_sm / num_terms;
    blocks_per_term = std::min(std::max(blocks_per_term, 4), 32);
    int total_blocks = num_terms * blocks_per_term;
    int needed_states = total_blocks * config.threads_per_block;

    if (!g_sobol_qmc.initialized || g_sobol_qmc.num_states < needed_states) {
        if (g_sobol_qmc.initialized) g_sobol_qmc.cleanup();

        g_sobol_qmc.init(needed_states, dimensions);

        dim3 init_blocks((needed_states + 255) / 256);
        dim3 init_threads(256);
        init_sobol_kernel<<<init_blocks, init_threads>>>(
            g_sobol_qmc.states, g_sobol_qmc.directions, dimensions, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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

    size_t block_sums_elems = (size_t)num_terms * blocks_per_term;
    double* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, block_sums_elems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, block_sums_elems * sizeof(double)));

    cudaStream_t streams[3];
    for(int i=0;i<3;i++) CUDA_CHECK(cudaStreamCreate(&streams[i]));

    std::vector<int> fp64_terms, fp32_terms, fp16_terms;
    for(int i=0;i<num_terms;i++){
        if(prec_i[i]==2) fp64_terms.push_back(i);
        else if(prec_i[i]==1) fp32_terms.push_back(i);
        else fp16_terms.push_back(i);
    }

    dim3 threads(config.threads_per_block);

    dim3 blocks(num_terms * blocks_per_term);

    if(!fp64_terms.empty()){
        mc_integrate_kernel_mixed<<<blocks, threads, 0, streams[0]>>>(
            d_types, d_constants, d_var_indices, d_op_codes, expr_length,
            d_bounds_min, d_bounds_max, dimensions,
            d_samples, num_terms, blocks_per_term, d_block_sums, d_precisions, g_sobol_qmc.states);
    }

    if(!fp32_terms.empty()){
        mc_integrate_kernel_mixed<<<blocks, threads, 0, streams[1]>>>(
            d_types, d_constants, d_var_indices, d_op_codes, expr_length,
            d_bounds_min, d_bounds_max, dimensions,
            d_samples, num_terms, blocks_per_term, d_block_sums, d_precisions, g_sobol_qmc.states);
    }

    if(!fp16_terms.empty()){
        mc_integrate_kernel_mixed<<<blocks, threads, 0, streams[2]>>>(
            d_types, d_constants, d_var_indices, d_op_codes, expr_length,
            d_bounds_min, d_bounds_max, dimensions,
            d_samples, num_terms, blocks_per_term, d_block_sums, d_precisions, g_sobol_qmc.states);
    }

    for(int i=0;i<3;i++) CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    for(int i=0;i<3;i++) CUDA_CHECK(cudaStreamDestroy(streams[i]));

    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_block_sums(block_sums_elems);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          block_sums_elems * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> results(num_terms);
    for (int i = 0; i < num_terms; ++i) {
        double sum = 0.0;
        for (int b = 0; b < blocks_per_term; ++b) {
            sum += h_block_sums[(size_t)i * blocks_per_term + b];
        }

        const double* bounds_min_term = &flat_min[i * dimensions];
        const double* bounds_max_term = &flat_max[i * dimensions];
        double volume = 1.0;
        for (int d = 0; d < dimensions; ++d) {
            volume *= (bounds_max_term[d] - bounds_min_term[d]);
        }

        results[i] = sum * volume / (double)samp_i[i];
    }

    cudaFree(d_types);
    cudaFree(d_constants);
    cudaFree(d_var_indices);
    cudaFree(d_op_codes);
    cudaFree(d_bounds_min);
    cudaFree(d_bounds_max);
    cudaFree(d_precisions);
    cudaFree(d_samples);
    cudaFree(d_block_sums);

    return results;
}

std::vector<double> monte_carlo_integrate_nd_cuda_mixed(
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& exprs,
    const std::vector<Precision>& precisions,
    const std::vector<size_t>& samples_per_term,
    const GPUConfig& config)
{
    std::vector<std::vector<double>> empty_bounds_min;
    std::vector<std::vector<double>> empty_bounds_max;

    return monte_carlo_integrate_nd_cuda_batch_mixed(
        0, bounds_min, bounds_max,
        empty_bounds_min, empty_bounds_max,
        samples_per_term, exprs, precisions, config
    );
}

// Explicit template instantiations
template std::vector<float> monte_carlo_integrate_nd_cuda_batch<float>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);

template std::vector<double> monte_carlo_integrate_nd_cuda_batch<double>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);
