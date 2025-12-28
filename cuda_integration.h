#ifndef CUDA_INTEGRATION_H
#define CUDA_INTEGRATION_H

#include <vector>
#include "parser.h"
#include "precision.h"

struct GPUConfig {
    int threads_per_block;
    int blocks_per_sm;
    int sm_count;

    GPUConfig() : threads_per_block(256), blocks_per_sm(4), sm_count(0) {}
};

GPUConfig detect_gpu();

// Initialize FP16 constant memory (call once at startup)
void init_fp16_constants();

// Template batch integration
template <typename T>
std::vector<T> monte_carlo_integrate_nd_cuda_batch(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& exprs,
    const GPUConfig& config);

// FP32 specialization
std::vector<float> monte_carlo_integrate_nd_cuda_batch_fp32(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& exprs,
    const GPUConfig& config);

// FP16 specialization
std::vector<float> monte_carlo_integrate_nd_cuda_batch_fp16(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& exprs,
    const GPUConfig& config);

// Mixed precision integration
std::vector<double> monte_carlo_integrate_nd_cuda_mixed(
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& exprs,
    const std::vector<Precision>& precisions,
    const std::vector<size_t>& samples_per_term,
    const GPUConfig& config);

// Mixed precision batch integration with per-term bounds
std::vector<double> monte_carlo_integrate_nd_cuda_batch_mixed(
    size_t samples,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<std::vector<double>>& bounds_min_per_term,
    const std::vector<std::vector<double>>& bounds_max_per_term,
    const std::vector<size_t>& samples_per_term,
    const std::vector<CompiledExpr>& exprs,
    const std::vector<Precision>& precisions,
    const GPUConfig& config);

// Explicit instantiation declarations
extern template std::vector<float> monte_carlo_integrate_nd_cuda_batch<float>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);

extern template std::vector<double> monte_carlo_integrate_nd_cuda_batch<double>(
    size_t, const std::vector<double>&, const std::vector<double>&,
    const std::vector<CompiledExpr>&, const GPUConfig&);

#endif
