#ifndef CUDA_INTEGRATION_H
#define CUDA_INTEGRATION_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include "parser.h"
#include "precision.h"

// GPU configuration for optimal performance
struct GPUConfig {
    int threads_per_block;
    int blocks_per_sm;
    int sm_count;
    
    GPUConfig() : threads_per_block(256), blocks_per_sm(4), sm_count(0) {}
};

GPUConfig get_optimal_gpu_config();

// Prepare compiled data for CUDA
void prepare_compiled_cuda_data(
    const CompiledExpr& compiled,
    TokenType** d_types,
    float** d_constants,
    int** d_var_indices,
    int** d_op_codes,
    int* expr_length);

// Multi-dimensional integration (1D-4D supported)
template <typename T>
std::vector<T> monte_carlo_integrate_nd_cuda_batch(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& all_compiled,
    const GPUConfig& config);

// FP16 batch integration (multi-dimensional)
std::vector<float> monte_carlo_integrate_nd_cuda_batch_fp16(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& all_compiled,
    const GPUConfig& config);

// Mixed-precision batch integration (per-term precision array).
// Returns double results (one per term). Bounds can be provided per-term by
// passing `bounds_min_per_term`/`bounds_max_per_term` (size = num_terms x dims).
std::vector<double> monte_carlo_integrate_nd_cuda_batch_mixed(
    size_t samples,
    const std::vector<double>& bounds_min, // if empty, use per-term bounds
    const std::vector<double>& bounds_max,
    const std::vector<std::vector<double>>& bounds_min_per_term,
    const std::vector<std::vector<double>>& bounds_max_per_term,
    const std::vector<size_t>& samples_per_term,
    const std::vector<CompiledExpr>& all_compiled,
    const std::vector<Precision>& precisions,
    const GPUConfig& config);

// Region-wise integration with adaptive partitioning
template <typename T>
std::vector<T> monte_carlo_integrate_regions_cuda_adaptive(
    const std::vector<Region>& regions,
    size_t samples_per_region,
    const CompiledExpr& compiled,
    const GPUConfig& config);

// --- v2-style REGION-WISE BATCH KERNEL (compiled expressions per-region)
template <typename T>
std::vector<T> monte_carlo_integrate_regions_cuda_batch_compiled(
    const std::vector<Region>& regions,
    size_t samples_per_region,
    const CompiledExpr& compiled);

extern template std::vector<float> monte_carlo_integrate_regions_cuda_batch_compiled<float>(
    const std::vector<Region>&, size_t, const CompiledExpr&);

extern template std::vector<double> monte_carlo_integrate_regions_cuda_batch_compiled<double>(
    const std::vector<Region>&, size_t, const CompiledExpr&);

// Explicit template declarations
extern template std::vector<float> monte_carlo_integrate_nd_cuda_batch<float>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);
    
extern template std::vector<double> monte_carlo_integrate_nd_cuda_batch<double>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);

extern template std::vector<float> monte_carlo_integrate_regions_cuda_adaptive<float>(
    const std::vector<Region>&, size_t, const CompiledExpr&, const GPUConfig&);
    
extern template std::vector<double> monte_carlo_integrate_regions_cuda_adaptive<double>(
    const std::vector<Region>&, size_t, const CompiledExpr&, const GPUConfig&);

#endif // CUDA_INTEGRATION_H