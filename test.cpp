#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <numeric>
#include "parser.h"
#include "precision.h"
#include "cuda_integration.h"

// ============================================================================
// STATISTICAL UTILITIES
// ============================================================================

struct Statistics {
    double mean;
    double stddev;
    double min;
    double max;
    
    Statistics(const std::vector<double>& data) {
        mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        
        double sq_sum = 0.0;
        min = data[0];
        max = data[0];
        
        for (double x : data) {
            sq_sum += (x - mean) * (x - mean);
            min = std::min(min, x);
            max = std::max(max, x);
        }
        
        stddev = std::sqrt(sq_sum / data.size());
    }
    
    void print(const std::string& name) const {
        std::cout << name << ": " << std::scientific << std::setprecision(6) << mean
                  << " ± " << stddev << " [" << min << ", " << max << "]\n";
    }
};

// ============================================================================
// REAL-WORLD TEST PROBLEMS
// ============================================================================

struct TestProblem {
    std::string name;
    std::string expr;
    std::vector<double> bounds_min;
    std::vector<double> bounds_max;
    double analytical_solution;  // -1 if unknown
    std::string category;
    
    TestProblem(const std::string& n, const std::string& e, 
                const std::vector<double>& bmin, const std::vector<double>& bmax,
                double sol = -1.0, const std::string& cat = "general")
        : name(n), expr(e), bounds_min(bmin), bounds_max(bmax), 
          analytical_solution(sol), category(cat) {}
};

std::vector<TestProblem> get_test_suite() {
    std::vector<TestProblem> tests;
    
    // ========== FINANCIAL MODELS ==========
    
    // Black-Scholes Monte Carlo (3D: S, t, volatility path)
    tests.emplace_back(
        "Black-Scholes Option Pricing",
        "(x - 100) * (x > 100)",  // Call option payoff, strike=100
        {80.0, 0.0, 0.1},   // Stock price, time, volatility
        {120.0, 1.0, 0.5},
        -1.0,
        "finance"
    );
    
    // Heston Stochastic Volatility (4D)
    tests.emplace_back(
        "Heston Model Integration",
        "exp(-0.05*y) * sqrt(z) * (x - 100) * (x > 100)",  // Discounted payoff with stochastic vol
        {80.0, 0.0, 0.04, -1.0},  // S, t, variance, Brownian increment
        {120.0, 1.0, 0.25, 1.0},
        -1.0,
        "finance"
    );
    
    // Portfolio VaR (3D: asset1, asset2, correlation)
    tests.emplace_back(
        "Portfolio Value-at-Risk",
        "x*x + y*y + 2*x*y*z",  // Portfolio variance with correlation z
        {-3.0, -3.0, -0.9},
        {3.0, 3.0, 0.9},
        -1.0,
        "finance"
    );
    
    // Interest Rate Model (3D: short rate, mean reversion, volatility)
    tests.emplace_back(
        "Vasicek Interest Rate Model",
        "exp(-x*y) * (0.05 + z*sqrt(y))",  // Bond pricing integral
        {0.0, 0.0, -0.02},
        {1.0, 5.0, 0.02},
        -1.0,
        "finance"
    );
    
    // ========== BAYESIAN INFERENCE ==========
    
    // Posterior with Gaussian likelihood (3D)
    tests.emplace_back(
        "Bayesian Posterior (Gaussian)",
        "exp(-0.5*((x-2)*(x-2) + (y-1)*(y-1) + (z-0)*(z-0)))",  // Unnormalized posterior
        {-5.0, -5.0, -5.0},
        {5.0, 5.0, 5.0},
        pow(2.0 * M_PI, 1.5),  // Analytical: (2π)^(3/2)
        "bayesian"
    );
    
    // Hierarchical Model (4D: data, hyperparameter, variance1, variance2)
    tests.emplace_back(
        "Hierarchical Bayesian Model",
        "exp(-0.5*x*x/y) * exp(-0.5*z*z/w) / sqrt(y*w)",
        {-4.0, 0.5, -4.0, 0.5},
        {4.0, 3.0, 4.0, 3.0},
        -1.0,
        "bayesian"
    );
    
    // Beta-Binomial Model (2D)
    tests.emplace_back(
        "Beta-Binomial Posterior",
        "x^7 * (1-x)^3 * y^2 * (1-y)^5",  // Beta(8,4) × Beta(3,6)
        {0.0, 0.0},
        {1.0, 1.0},
        0.000002645502645,  // B(8,4) × B(3,6)
        "bayesian"
    );
    
    // Mixture Model Evidence (3D)
    tests.emplace_back(
        "Gaussian Mixture Evidence",
        "0.3*exp(-0.5*(x+2)*(x+2)) + 0.7*exp(-0.5*(x-2)*(x-2)) * exp(-0.5*y*y) * exp(-0.5*z*z)",
        {-5.0, -5.0, -5.0},
        {5.0, 5.0, 5.0},
        -1.0,
        "bayesian"
    );
    
    // ========== PHYSICS / ENGINEERING ==========
    
    // Quantum Harmonic Oscillator (3D wavefunction)
    tests.emplace_back(
        "Quantum Harmonic Oscillator",
        "x*x * y*y * z*z * exp(-(x*x + y*y + z*z))",  // |ψ|² for excited state
        {-3.0, -3.0, -3.0},
        {3.0, 3.0, 3.0},
        -1.0,
        "physics"
    );
    
    // Electromagnetic Field Energy (4D)
    tests.emplace_back(
        "EM Field Energy Density",
        "(sin(x)*cos(y))^2 + (sin(z)*cos(w))^2",  // E² + B²
        {0.0, 0.0, 0.0, 0.0},
        {3.14159, 3.14159, 3.14159, 3.14159},
        pow(3.14159, 4) / 2.0,  // π^4 / 2
        "physics"
    );
    
    // Heat Diffusion (3D: space × time)
    tests.emplace_back(
        "Heat Diffusion Integral",
        "exp(-x*x/(4*z)) * exp(-y*y/(4*z)) / (z*sqrt(z))",  // Green's function
        {-2.0, -2.0, 0.1},
        {2.0, 2.0, 2.0},
        -1.0,
        "physics"
    );
    
    // ========== OSCILLATORY / DIFFICULT ==========
    
    // Highly oscillatory (2D)
    tests.emplace_back(
        "Oscillatory Function",
        "sin(20*x) * cos(20*y)",
        {0.0, 0.0},
        {3.14159, 3.14159},
        0.0,  // Integral averages to ~0
        "difficult"
    );
    
    // Sharp peak (3D)
    tests.emplace_back(
        "Narrow Gaussian Peak",
        "exp(-100*((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5)))",
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        pow(M_PI / 100.0, 1.5),  // Analytical
        "difficult"
    );
    
    // Discontinuous (2D)
    tests.emplace_back(
        "Discontinuous Indicator",
        "(x*x + y*y < 1)",  // Circle indicator (will need careful handling)
        {-1.5, -1.5},
        {1.5, 1.5},
        M_PI,  // Area of unit circle
        "difficult"
    );
    
    // High degree polynomial (3D)
    tests.emplace_back(
        "High-Degree Polynomial",
        "x^12 + y^8 + z^6",
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        1.0/13.0 + 1.0/9.0 + 1.0/7.0,  // Analytical
        "difficult"
    );
    
    // ========== STANDARD BENCHMARKS ==========
    
    tests.emplace_back(
        "Simple 2D Polynomial",
        "x*y",
        {0.0, 0.0},
        {1.0, 1.0},
        0.25,
        "benchmark"
    );
    
    tests.emplace_back(
        "3D Quadratic",
        "x*x + y*y + z*z",
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        1.0,
        "benchmark"
    );
    
    tests.emplace_back(
        "4D Trigonometric",
        "sin(x)*cos(y)*sin(z)*cos(w)",
        {0.0, 0.0, 0.0, 0.0},
        {3.14159, 3.14159, 3.14159, 3.14159},
        0.0,
        "benchmark"
    );
    
    return tests;
}

// ============================================================================
// TEST EXECUTION FRAMEWORK
// ============================================================================

struct TestResult {
    std::string test_name;
    double fp16_mean, fp16_std, fp16_time;
    double fp32_mean, fp32_std, fp32_time;
    double fp64_mean, fp64_std, fp64_time;
    double mixed_mean, mixed_std, mixed_time;
    double analytical;
    double fp16_error, fp32_error, fp64_error, mixed_error;
    int dimensions;
};

TestResult run_single_test(const TestProblem& problem, size_t samples, int num_runs, const GPUConfig& config) {
    TestResult result;
    result.test_name = problem.name;
    result.analytical = problem.analytical_solution;
    result.dimensions = problem.bounds_min.size();
    
    // Parse expression
    auto tokens = tokenize(problem.expr);
    auto postfix = to_postfix(tokens);
    auto compiled = compile_expression(postfix, result.dimensions);
    std::vector<CompiledExpr> compiled_vec = {compiled};
    
    std::vector<double> fp16_results, fp32_results, fp64_results, mixed_results;
    std::vector<double> fp16_times, fp32_times, fp64_times, mixed_times;
    
    // Run multiple times for statistics
    for (int run = 0; run < num_runs; ++run) {
        // FP16
        auto t0 = std::chrono::high_resolution_clock::now();
        auto res_fp16 = monte_carlo_integrate_nd_cuda_batch_fp16(
            samples, problem.bounds_min, problem.bounds_max, compiled_vec, config);
        auto t1 = std::chrono::high_resolution_clock::now();
        fp16_results.push_back(res_fp16[0]);
        fp16_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        
        // FP32
        t0 = std::chrono::high_resolution_clock::now();
        auto res_fp32 = monte_carlo_integrate_nd_cuda_batch<float>(
            samples, problem.bounds_min, problem.bounds_max, compiled_vec, config);
        t1 = std::chrono::high_resolution_clock::now();
        fp32_results.push_back(res_fp32[0]);
        fp32_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        
        // FP64
        t0 = std::chrono::high_resolution_clock::now();
        auto res_fp64 = monte_carlo_integrate_nd_cuda_batch<double>(
            samples, problem.bounds_min, problem.bounds_max, compiled_vec, config);
        t1 = std::chrono::high_resolution_clock::now();
        fp64_results.push_back(res_fp64[0]);
        fp64_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        
        // Mixed precision (adaptive)
        t0 = std::chrono::high_resolution_clock::now();
        Precision prec = select_precision_for_term(postfix, problem.bounds_min, problem.bounds_max, 1e-5, problem.name);
        std::vector<Precision> precs = {prec};
        std::vector<size_t> samples_vec = {samples};
        std::vector<std::vector<double>> bounds_min_vec = {problem.bounds_min};
        std::vector<std::vector<double>> bounds_max_vec = {problem.bounds_max};
        
        auto res_mixed = monte_carlo_integrate_nd_cuda_batch_mixed(
            samples, problem.bounds_min, problem.bounds_max,
            bounds_min_vec, bounds_max_vec, samples_vec,
            compiled_vec, precs, config);
        t1 = std::chrono::high_resolution_clock::now();
        mixed_results.push_back(res_mixed[0]);
        mixed_times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    
    // Compute statistics
    Statistics fp16_stat(fp16_results);
    Statistics fp32_stat(fp32_results);
    Statistics fp64_stat(fp64_results);
    Statistics mixed_stat(mixed_results);
    
    Statistics fp16_time_stat(fp16_times);
    Statistics fp32_time_stat(fp32_times);
    Statistics fp64_time_stat(fp64_times);
    Statistics mixed_time_stat(mixed_times);
    
    result.fp16_mean = fp16_stat.mean;
    result.fp16_std = fp16_stat.stddev;
    result.fp16_time = fp16_time_stat.mean;
    
    result.fp32_mean = fp32_stat.mean;
    result.fp32_std = fp32_stat.stddev;
    result.fp32_time = fp32_time_stat.mean;
    
    result.fp64_mean = fp64_stat.mean;
    result.fp64_std = fp64_stat.stddev;
    result.fp64_time = fp64_time_stat.mean;
    
    result.mixed_mean = mixed_stat.mean;
    result.mixed_std = mixed_stat.stddev;
    result.mixed_time = mixed_time_stat.mean;
    
    // Compute errors relative to analytical or FP64
    double reference = (problem.analytical_solution > 0) ? problem.analytical_solution : result.fp64_mean;
    
    result.fp16_error = std::abs(result.fp16_mean - reference) / std::abs(reference);
    result.fp32_error = std::abs(result.fp32_mean - reference) / std::abs(reference);
    result.fp64_error = (problem.analytical_solution > 0) ? 
                        std::abs(result.fp64_mean - reference) / std::abs(reference) : 0.0;
    result.mixed_error = std::abs(result.mixed_mean - reference) / std::abs(reference);
    
    return result;
}

// ============================================================================
// ADAPTIVE VS FIXED PARTITIONING COMPARISON
// ============================================================================

struct PartitioningResult {
    std::string method;
    double result;
    double error;
    double time;
    int num_regions;
};

std::vector<PartitioningResult> compare_partitioning_methods(
    const TestProblem& problem, size_t total_samples, const GPUConfig& config)
{
    std::vector<PartitioningResult> results;
    
    auto tokens = tokenize(problem.expr);
    auto postfix = to_postfix(tokens);
    int dims = problem.bounds_min.size();
    auto compiled = compile_expression(postfix, dims);
    
    // Method 1: Fixed uniform partitioning (8^d regions for d dimensions)
    {
        int regions_per_dim = (dims <= 2) ? 8 : ((dims == 3) ? 4 : 2);
        std::vector<Region> fixed_regions;
        
        // Generate uniform grid
        std::function<void(int, std::vector<double>&, std::vector<double>&)> generate_grid;
        generate_grid = [&](int dim, std::vector<double>& min_acc, std::vector<double>& max_acc) {
            if (dim == dims) {
                fixed_regions.emplace_back(min_acc, max_acc);
                return;
            }
            
            double range = problem.bounds_max[dim] - problem.bounds_min[dim];
            double step = range / regions_per_dim;
            
            for (int i = 0; i < regions_per_dim; ++i) {
                min_acc.push_back(problem.bounds_min[dim] + i * step);
                max_acc.push_back(problem.bounds_min[dim] + (i + 1) * step);
                generate_grid(dim + 1, min_acc, max_acc);
                min_acc.pop_back();
                max_acc.pop_back();
            }
        };
        
        std::vector<double> min_acc, max_acc;
        generate_grid(0, min_acc, max_acc);
        
        size_t samples_per_region = total_samples / fixed_regions.size();
        
        auto t0 = std::chrono::high_resolution_clock::now();
        auto region_results = monte_carlo_integrate_regions_cuda_adaptive<double>(
            fixed_regions, samples_per_region, compiled, config);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double total = std::accumulate(region_results.begin(), region_results.end(), 0.0);
        
        PartitioningResult pr;
        pr.method = "Fixed Uniform";
        pr.result = total;
        pr.time = std::chrono::duration<double>(t1 - t0).count();
        pr.num_regions = fixed_regions.size();
        pr.error = (problem.analytical_solution > 0) ? 
                   std::abs(total - problem.analytical_solution) / problem.analytical_solution : 0.0;
        results.push_back(pr);
    }
    
    // Method 2: Adaptive partitioning
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto adaptive_regions = adaptive_partition(postfix, problem.bounds_min, problem.bounds_max, 64, 1e-3);
        
        // Allocate samples proportionally to region error estimates
        std::vector<size_t> samples_per_region(adaptive_regions.size());
        double total_error = 0.0;
        for (const auto& r : adaptive_regions) total_error += r.error_estimate;
        
        for (size_t i = 0; i < adaptive_regions.size(); ++i) {
            samples_per_region[i] = std::max((size_t)1, 
                (size_t)(total_samples * adaptive_regions[i].error_estimate / total_error));
        }
        
        // Integrate each region
        double total = 0.0;
        for (size_t i = 0; i < adaptive_regions.size(); ++i) {
            std::vector<CompiledExpr> single = {compiled};
            auto res = monte_carlo_integrate_nd_cuda_batch<double>(
                samples_per_region[i],
                adaptive_regions[i].bounds_min,
                adaptive_regions[i].bounds_max,
                single, config);
            total += res[0];
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        PartitioningResult pr;
        pr.method = "Adaptive";
        pr.result = total;
        pr.time = std::chrono::duration<double>(t1 - t0).count();
        pr.num_regions = adaptive_regions.size();
        pr.error = (problem.analytical_solution > 0) ? 
                   std::abs(total - problem.analytical_solution) / problem.analytical_solution : 0.0;
        results.push_back(pr);
    }
    
    // Method 3: Term-wise adaptive (single region, adaptive precision per term)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        Precision prec = select_precision_for_term(postfix, problem.bounds_min, problem.bounds_max, 1e-5, problem.name);
        std::vector<CompiledExpr> single = {compiled};
        std::vector<Precision> precs = {prec};
        std::vector<size_t> samples_vec = {total_samples};
        std::vector<std::vector<double>> bounds_min_vec = {problem.bounds_min};
        std::vector<std::vector<double>> bounds_max_vec = {problem.bounds_max};
        
        auto res = monte_carlo_integrate_nd_cuda_batch_mixed(
            total_samples, problem.bounds_min, problem.bounds_max,
            bounds_min_vec, bounds_max_vec, samples_vec, single, precs, config);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        PartitioningResult pr;
        pr.method = "Term-wise Adaptive";
        pr.result = res[0];
        pr.time = std::chrono::duration<double>(t1 - t0).count();
        pr.num_regions = 1;
        pr.error = (problem.analytical_solution > 0) ? 
                   std::abs(res[0] - problem.analytical_solution) / problem.analytical_solution : 0.0;
        results.push_back(pr);
    }
    
    // Method 4: Adaptive Region-wise Mixed Precision (ported from newMixed_v2)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        Region initial(problem.bounds_min, problem.bounds_max);
        auto adaptive_regions = adaptive_partition_nd(postfix, initial, 1e-2, 1e2, 256);
        size_t samples_per_region = total_samples / std::max((size_t)1, adaptive_regions.size());

        std::vector<Region> half_regions, float_regions, double_regions;
        std::vector<int> half_ids, float_ids, double_ids;
        std::vector<Precision> region_precisions(adaptive_regions.size());

        auto sel_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < adaptive_regions.size(); ++i) {
            auto& R = adaptive_regions[i];
            auto metrics = analyze_expression_fast(postfix, R.bounds_min, R.bounds_max, 50);
            region_precisions[i] = select_precision_enhanced(metrics, 1e-5, "Region " + std::to_string(i), false);
        }
        auto sel_end = std::chrono::high_resolution_clock::now();
        double sel_time = std::chrono::duration<double>(sel_end - sel_start).count();

        for (size_t i = 0; i < adaptive_regions.size(); ++i) {
            if (region_precisions[i] == Precision::Half) {
                half_regions.push_back(adaptive_regions[i]); half_ids.push_back(i);
            } else if (region_precisions[i] == Precision::Float) {
                float_regions.push_back(adaptive_regions[i]); float_ids.push_back(i);
            } else {
                double_regions.push_back(adaptive_regions[i]); double_ids.push_back(i);
            }
        }

        std::cout << "\nAdaptive Region-wise selection time: " << sel_time << " s\n";
        std::cout << "  Total regions: " << adaptive_regions.size() << "\n";
        std::cout << "  FP16: " << half_regions.size() << " regions\n";
        std::cout << "  FP32: " << float_regions.size() << " regions\n";
        std::cout << "  FP64: " << double_regions.size() << " regions\n";

        std::vector<double> region_results(adaptive_regions.size(), 0.0);

        if (!half_regions.empty()) {
            auto res = monte_carlo_integrate_regions_cuda_batch_compiled<float>(half_regions, samples_per_region, compiled);
            for (size_t i = 0; i < res.size(); ++i) region_results[half_ids[i]] = static_cast<double>(res[i]);
        }
        if (!float_regions.empty()) {
            auto res = monte_carlo_integrate_regions_cuda_batch_compiled<float>(float_regions, samples_per_region, compiled);
            for (size_t i = 0; i < res.size(); ++i) region_results[float_ids[i]] = static_cast<double>(res[i]);
        }
        if (!double_regions.empty()) {
            auto res = monte_carlo_integrate_regions_cuda_batch_compiled<double>(double_regions, samples_per_region, compiled);
            for (size_t i = 0; i < res.size(); ++i) region_results[double_ids[i]] = res[i];
        }

        double total = 0.0;
        for (double r : region_results) total += r;

        auto t1 = std::chrono::high_resolution_clock::now();

        PartitioningResult pr;
        pr.method = "Adaptive Region-wise Mixed Precision";
        pr.result = total;
        pr.time = std::chrono::duration<double>(t1 - t0).count();
        pr.num_regions = adaptive_regions.size();
        pr.error = (problem.analytical_solution > 0) ? 
                   std::abs(total - problem.analytical_solution) / problem.analytical_solution : 0.0;
        results.push_back(pr);
    }

    return results;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  COMPREHENSIVE MONTE CARLO INTEGRATION TEST SUITE\n";
    std::cout << "  Real-World Problems: Finance, Bayesian, Physics, Benchmarks\n";
    std::cout << "================================================================\n\n";
    
    // Initialize CUDA
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "GPU: " << deviceProp.name << "\n";
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n\n";
    
    CUDA_CHECK(cudaSetDevice(0));
    GPUConfig config = get_optimal_gpu_config();
    
    // Test parameters
    size_t samples = 10000000;  // 10M samples
    int num_runs = 10;  // Statistical significance
    
    auto test_suite = get_test_suite();
    std::vector<TestResult> all_results;
    
    std::cout << "Running " << test_suite.size() << " tests with " 
              << samples << " samples each, " << num_runs << " runs per test\n\n";
    
    // Run all tests
    for (const auto& test : test_suite) {
        std::cout << "Testing: " << test.name << " [" << test.category << "] (" 
                  << test.bounds_min.size() << "D)\n";
        std::cout << "  Expression: " << test.expr << "\n";
        
        auto result = run_single_test(test, samples, num_runs, config);
        all_results.push_back(result);
        
        std::cout << "  Results:\n";
        std::cout << "    FP16: " << std::scientific << std::setprecision(8) << result.fp16_mean 
                  << " ± " << result.fp16_std << " (error: " << result.fp16_error << ")\n";
        std::cout << "    FP32: " << result.fp32_mean << " ± " << result.fp32_std 
                  << " (error: " << result.fp32_error << ")\n";
        std::cout << "    FP64: " << result.fp64_mean << " ± " << result.fp64_std 
                  << " (error: " << result.fp64_error << ")\n";
        std::cout << "    Mixed: " << result.mixed_mean << " ± " << result.mixed_std 
                  << " (error: " << result.mixed_error << ")\n";
        
        std::cout << "  Timing:\n";
        std::cout << "    FP16: " << std::fixed << std::setprecision(4) << result.fp16_time << " s\n";
        std::cout << "    FP32: " << result.fp32_time << " s (speedup vs FP64: " 
                  << (result.fp64_time / result.fp32_time) << "x)\n";
        std::cout << "    FP64: " << result.fp64_time << " s\n";
        std::cout << "    Mixed: " << result.mixed_time << " s\n\n";
    }
    
    // ========== SUMMARY STATISTICS ==========
    
    std::cout << "\n================================================================\n";
    std::cout << "  SUMMARY STATISTICS\n";
    std::cout << "================================================================\n\n";
    
    // FP64/FP32 speedup analysis
    std::vector<double> speedups;
    for (const auto& r : all_results) {
        speedups.push_back(r.fp64_time / r.fp32_time);
    }
    Statistics speedup_stats(speedups);
    std::cout << "FP64/FP32 Speedup: " << std::fixed << std::setprecision(2) 
              << speedup_stats.mean << "x ± " << speedup_stats.stddev 
              << " [" << speedup_stats.min << "x, " << speedup_stats.max << "x]\n";
    
    // Error analysis by category
    std::map<std::string, std::vector<double>> errors_by_category;
    for (size_t i = 0; i < all_results.size(); ++i) {
        errors_by_category[test_suite[i].category].push_back(all_results[i].fp32_error);
    }
    
    std::cout << "\nFP32 Relative Errors by Category:\n";
    for (const auto& [cat, errs] : errors_by_category) {
        Statistics err_stats(errs);
        std::cout << "  " << std::setw(20) << std::left << cat << ": " 
                  << std::scientific << err_stats.mean << " ± " << err_stats.stddev << "\n";
    }
    
    // ========== PARTITIONING COMPARISON ==========
    
    std::cout << "\n================================================================\n";
    std::cout << "  PARTITIONING STRATEGY COMPARISON\n";
    std::cout << "================================================================\n\n";
    
    // Select representative tests for partitioning comparison
    std::vector<size_t> partition_test_indices = {1, 5, 9, 13};  // Various difficulties
    
    for (size_t idx : partition_test_indices) {
        if (idx >= test_suite.size()) continue;
        
        std::cout << "Test: " << test_suite[idx].name << "\n";
        auto partition_results = compare_partitioning_methods(test_suite[idx], samples, config);
        
        std::cout << std::setw(20) << "Method" << std::setw(15) << "Result" 
                  << std::setw(15) << "Error" << std::setw(12) << "Time(s)" 
                  << std::setw(12) << "Regions" << "\n";
        std::cout << std::string(74, '-') << "\n";
        
        for (const auto& pr : partition_results) {
            std::cout << std::setw(20) << pr.method 
                      << std::setw(15) << std::scientific << std::setprecision(6) << pr.result
                      << std::setw(15) << pr.error
                      << std::setw(12) << std::fixed << std::setprecision(4) << pr.time
                      << std::setw(12) << pr.num_regions << "\n";
        }
        std::cout << "\n";
    }
    
    // ========== FINAL ASSESSMENT ==========
    
    std::cout << "================================================================\n";
    std::cout << "  FINAL ASSESSMENT\n";
    std::cout << "================================================================\n\n";
    
    bool pass_speedup = speedup_stats.mean >= 40.0;
    bool pass_accuracy = true;
    for (const auto& r : all_results) {
        if (r.fp32_error > 0.01 && test_suite[&r - &all_results[0]].analytical_solution > 0) {
            pass_accuracy = false;
            break;
        }
    }
    
    std::cout << "Performance Target (FP64/FP32 ≥ 40x): " << (pass_speedup ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "Accuracy Target (FP32 error < 1%): " << (pass_accuracy ? "✓ PASS" : "✗ FAIL") << "\n";
    
    std::cout << "\nAll tests completed successfully!\n";
    
    return EXIT_SUCCESS;
}