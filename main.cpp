#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "parser.h"
#include "precision.h"
#include "cuda_integration.h"
#include <sstream>
#include <set>
#include <regex>

// ============================================================================
// COMPREHENSIVE ACCURACY ANALYSIS - FIXED VERSION
// ============================================================================

void analyze_accuracy_comprehensive(
    const std::vector<std::string>& terms,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    const std::vector<CompiledExpr>& compiled,
    const GPUConfig& config,
    bool enable_fp16)
{
    std::cout << "\n=== COMPREHENSIVE ACCURACY ANALYSIS ===\n";
    std::cout << "Testing COMPLETE expression with " << terms.size() << " terms\n\n";
    
    // Compute high-precision reference using ALL terms
    std::cout << "Computing FP64 reference (100M samples)...\n";
    auto ref_results = monte_carlo_integrate_nd_cuda_batch<double>(
        100000000, bounds_min, bounds_max, compiled, config);
    
    // Sum all terms for total result
    double analytical = 0.0;
    for (size_t i = 0; i < ref_results.size(); ++i) {
        analytical += ref_results[i];
    }
    
    std::cout << "FP64 reference (sum of all terms): " << std::scientific 
              << std::setprecision(12) << analytical << "\n";
    std::cout << "Individual term contributions:\n";
    for (size_t i = 0; i < std::min(size_t(5), ref_results.size()); ++i) {
        std::cout << "  Term " << i << " [" << terms[i] << "]: " 
                  << ref_results[i] << "\n";
    }
    if (ref_results.size() > 5) {
        std::cout << "  ... (" << (ref_results.size() - 5) << " more terms)\n";
    }
    std::cout << "\n";
    
    // Test multiple sample counts
    std::vector<size_t> sample_counts = {1000, 10000, 100000, 1000000, 10000000};

    std::cout << std::setw(12) << "Samples";
    if (enable_fp16) std::cout << std::setw(18) << "FP16 Rel.Error";
    std::cout << std::setw(18) << "FP32 Rel.Error"
              << std::setw(18) << "FP64 Rel.Error";
    if (enable_fp16) std::cout << std::setw(18) << "FP16 Value";
    std::cout << std::setw(18) << "FP32 Value"
              << std::setw(18) << "FP64 Value" << "\n";
    std::cout << std::string(enable_fp16 ? 120 : 84, '-') << "\n";
    
    struct Result { 
        size_t samples; 
        double fp16_err, fp32_err, fp64_err; 
        double fp16_val, fp32_val, fp64_val; 
    };
    std::vector<Result> results;
    
    for (size_t n : sample_counts) {
        Result r;
        r.samples = n;

        // FP16 - sum all terms (if enabled)
        if (enable_fp16) {
            auto res_fp16 = monte_carlo_integrate_nd_cuda_batch_fp16(
                n, bounds_min, bounds_max, compiled, config);
            r.fp16_val = 0.0;
            for (auto v : res_fp16) r.fp16_val += v;
            r.fp16_err = std::abs(r.fp16_val - analytical) / std::abs(analytical);
        } else {
            r.fp16_val = 0.0;
            r.fp16_err = 0.0;
        }

        // FP32 - sum all terms
        auto res_fp32 = monte_carlo_integrate_nd_cuda_batch<float>(
            n, bounds_min, bounds_max, compiled, config);
        r.fp32_val = 0.0;
        for (auto v : res_fp32) r.fp32_val += v;
        r.fp32_err = std::abs(r.fp32_val - analytical) / std::abs(analytical);

        // FP64 - sum all terms
        auto res_fp64 = monte_carlo_integrate_nd_cuda_batch<double>(
            n, bounds_min, bounds_max, compiled, config);
        r.fp64_val = 0.0;
        for (auto v : res_fp64) r.fp64_val += v;
        r.fp64_err = std::abs(r.fp64_val - analytical) / std::abs(analytical);

        std::cout << std::setw(12) << n;
        if (enable_fp16) std::cout << std::setw(18) << std::scientific << std::setprecision(4) << r.fp16_err;
        std::cout << std::setw(18) << std::scientific << std::setprecision(4) << r.fp32_err
                  << std::setw(18) << r.fp64_err;
        if (enable_fp16) std::cout << std::setw(18) << std::fixed << std::setprecision(8) << r.fp16_val;
        std::cout << std::setw(18) << std::fixed << std::setprecision(8) << r.fp32_val
                  << std::setw(18) << std::setprecision(10) << r.fp64_val << "\n";

        results.push_back(r);
    }
    
    // Convergence analysis
    std::cout << "\n=== Error Convergence Analysis ===\n";
    std::cout << "Using: Sobol QMC + Antithetic Variates (2x variance reduction)\n";
    std::cout << "Expected: O((log N)^d / N) for Sobol vs O(1/sqrt(N)) for pure MC\n";
    std::cout << "Note: Different precisions use SAME Sobol sequence (fair comparison)\n\n";

    if (results.size() >= 2) {
        double n1 = results[results.size()-2].samples;
        double n2 = results[results.size()-1].samples;
        double mc_theoretical = std::sqrt(n2 / n1);  // Pure MC
        double qmc_theoretical = (n2 / n1);           // Ideal QMC in low dims

        // Calculate actual improvement for each precision
        auto calc_improvement = [](double err1, double err2) {
            if (err2 < 1e-10 || err1 < 1e-10) return -1.0;  // Mark as N/A
            return err1 / err2;
        };

        double fp16_imp = calc_improvement(results[results.size()-2].fp16_err,
                                          results[results.size()-1].fp16_err);
        double fp32_imp = calc_improvement(results[results.size()-2].fp32_err,
                                          results[results.size()-1].fp32_err);
        double fp64_imp = calc_improvement(results[results.size()-2].fp64_err,
                                          results[results.size()-1].fp64_err);

        std::cout << "Sample increase: " << (int)n1 << " → " << (int)n2 << " (10x)\n";
        std::cout << "Theoretical convergence rates:\n";
        std::cout << "  Pure MC:      " << std::fixed << std::setprecision(2) << mc_theoretical << "x improvement\n";
        std::cout << "  Ideal QMC:    " << qmc_theoretical << "x improvement (low-dim limit)\n";
        std::cout << "  Sobol (9D):   ~5-8x improvement (dimension penalty)\n\n";

        std::cout << "Actual error improvements:\n";
        if (enable_fp16 && fp16_imp > 0) {
            std::cout << "  FP16: " << std::setprecision(2) << fp16_imp << "x";
            if (fp16_imp >= 5.0) std::cout << " ✓ (good QMC convergence)";
            else if (fp16_imp >= mc_theoretical) std::cout << " ~ (better than MC)";
            else std::cout << " ✗ (sampling noise dominates)";
            std::cout << "\n";
        }
        if (fp32_imp > 0) {
            std::cout << "  FP32: " << std::setprecision(2) << fp32_imp << "x";
            if (fp32_imp >= 5.0) std::cout << " ✓ (good QMC convergence)";
            else if (fp32_imp >= mc_theoretical) std::cout << " ~ (better than MC)";
            else std::cout << " ✗ (sampling noise dominates)";
            std::cout << "\n";
        }
        if (fp64_imp > 0) {
            std::cout << "  FP64: " << std::setprecision(2) << fp64_imp << "x";
            if (fp64_imp >= 5.0) std::cout << " ✓ (good QMC convergence)";
            else if (fp64_imp >= mc_theoretical) std::cout << " ~ (better than MC)";
            else std::cout << " ✗ (sampling noise dominates)";
            std::cout << "\n";
        }

        // Diagnosis
        bool poor_convergence = (fp32_imp > 0 && fp32_imp < mc_theoretical) ||
                               (fp64_imp > 0 && fp64_imp < mc_theoretical);

        if (poor_convergence) {
            std::cout << "\n⚠ WARNING: Poor convergence detected!\n";
            std::cout << "  Possible causes:\n";
            std::cout << "  1. High dimensionality (9D) reduces QMC effectiveness\n";
            std::cout << "  2. Sharp features or discontinuities in integrand\n";
            std::cout << "  3. Numerical cancellation dominating (e.g., T7, T8, T10)\n";
            std::cout << "  Recommendation: Use more samples or dimension reduction\n";
        }
    }
}

// ============================================================================
// PERFORMANCE METRICS
// ============================================================================

struct PerformanceMetrics {
    double kernel_time_ms;
    double throughput_msamples_per_sec;
    double gflops;
    
    void print() const {
        std::cout << "  Kernel time: " << std::fixed << std::setprecision(6) 
                  << (kernel_time_ms / 1000.0) << " s\n";
        std::cout << "  Throughput: " << std::setprecision(2) 
                  << throughput_msamples_per_sec << " MSamples/s\n";
        std::cout << "  Est. GFLOPS: " << std::setprecision(2) << gflops << "\n";
    }
};

PerformanceMetrics calculate_metrics(double time_ms, size_t samples, int flops = 15) {
    PerformanceMetrics m;
    m.kernel_time_ms = time_ms;
    m.throughput_msamples_per_sec = (samples / 1e6) / (time_ms / 1000.0);
    m.gflops = (samples * flops) / (time_ms * 1e6);
    return m;
}

void compare_speedups(double t16, double t32, double t64) {
    std::cout << "\n=== SPEEDUP ANALYSIS ===\n";
    if (t32 > 0.0) {
        std::cout << "FP32 is " << std::fixed << std::setprecision(2)
                  << (t64 / t32) << "x faster than FP64\n";
    }
    if (t16 > 0.0 && t32 > 0.0) {
        std::cout << "FP16 is " << (t32 / t16) << "x faster than FP32\n";
        std::cout << "FP16 is " << (t64 / t16) << "x faster than FP64\n";
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================



static std::vector<double> parse_comma_doubles(const std::string &s) {
    std::vector<double> out;
    std::istringstream iss(s);
    std::string part;
    while (std::getline(iss, part, ',')) {
        try {
            if (part.size()==0) continue;
            out.push_back(std::stod(part));
        } catch (...) {}
    }
    return out;
}

// Parse bounds given in several convenient formats:
// 1) "min1,min2,...;max1,max2,..."  (semicolon separates mins and maxs)
// 2) "min1:max1,min2:max2,..."      (per-dimension pairs)
// 3) "min:max"                       (1D shorthand)
static bool parse_bounds_string(const std::string &s, std::vector<double> &mins, std::vector<double> &maxs) {
    mins.clear(); maxs.clear();
    // Case 1: semicolon-separated mins and maxs
    auto pos = s.find(';');
    if (pos != std::string::npos) {
        std::string left = s.substr(0, pos);
        std::string right = s.substr(pos+1);
        auto l = parse_comma_doubles(left);
        auto r = parse_comma_doubles(right);
        if (l.empty() || r.empty()) return false;
        mins = l; maxs = r;
        return true;
    }

    // Case 2: comma-separated per-dimension pairs using ':'
    // e.g. "0:1,0:2"
    bool has_colon = (s.find(':') != std::string::npos);
    if (has_colon) {
        std::istringstream iss(s);
        std::string part;
        while (std::getline(iss, part, ',')) {
            auto p = part.find(':');
            if (p == std::string::npos) return false;
            try {
                double a = std::stod(part.substr(0, p));
                double b = std::stod(part.substr(p+1));
                mins.push_back(a);
                maxs.push_back(b);
            } catch (...) { return false; }
        }
        if (mins.empty() || maxs.empty()) return false;
        return true;
    }

    // Case 3: simple two numbers (min:max or min,max) for 1D
    auto parts = parse_comma_doubles(s);
    if (parts.size() == 2) {
        mins.push_back(parts[0]);
        maxs.push_back(parts[1]);
        return true;
    }

    // Couldn't parse
    return false;
}

int main(int argc, char** argv) {
    std::cout << "================================================================\n";
    std::cout << "  OPTIMIZED MIXED PRECISION MONTE CARLO INTEGRATION\n";
    std::cout << "  Features: FP16/FP32/FP64, Auto-dimension, Adaptive regions\n";
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
    std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
    std::cout << "  SMs: " << deviceProp.multiProcessorCount << "\n";
    std::cout << "  FP32 cores: ~" << (deviceProp.multiProcessorCount * 128) << "\n";
    std::cout << "  FP64 throughput: " << (deviceProp.major >= 8 ? "1/32" : "1/64") << " of FP32\n\n";
    
    CUDA_CHECK(cudaSetDevice(0));
    GPUConfig config = detect_gpu();

    // Initialize FP16 constant memory
    init_fp16_constants();

    // Defaults (used if not overridden via CLI)
    //std::string expr = "sin(x + y + z + w)+ cos(x*y) - log(1 + z*w) * x^5 + y^4 * exp(-w) - z^2 + x^12 - w^3 + sin(z*w^2) - 4";
    std::string expr =
    "sin(6.283185307179586*(a + b))"                                      // T1 oscillatory
    " + cos(5.0*c*d)"                                                    // T2 nonlinear coupling
    " + exp(-0.5*(x*x + v*v))"                                           // T3 Gaussian (statistics)
    " + exp(-abs(y))"                                                    // T4 Laplace / sparsity
    " + 1.0/(1.0 + exp(-25.0*(z - 0.05)))"                               // T5 logistic (ML / stat mech)
    " + log(1.0 + abs(a*x))"                                             // T6 logarithmic nonlinearity
    " + 0.01*(sqrt(1.0 + 0.00001*(b*c + d*x)) - 1.0)"                    // T7 FP64: catastrophic cancellation
    " + (log(1.0 + 0.00001*(v + y)) - 0.00001*(v + y))"                  // T8 FP64: log(1+u) − u
    " + 0.001 / sqrt(0.000000000001 + (y-0.02)*(y-0.02) + (z+0.03)*(z+0.03))"     // T9 FP64: sharp peak
    " + (exp(8.0*a) - exp(8.0*a - 0.00001))";                            // T10 FP64: exp cancellation

    size_t total_samples = 100000000; // 100M
    double tolerance = 1e-3;
    std::vector<double> bounds_min;
    std::vector<double> bounds_max;
    int explicit_dims = -1;
    bool enable_fp16 = false;

    // Simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--expr" && i+1 < argc) { expr = argv[++i]; }
        else if (a == "--func" && i+1 < argc) { expr = argv[++i]; }
        else if (a == "--samples" && i+1 < argc) { total_samples = std::stoull(argv[++i]); }
        else if (a == "--sample" && i+1 < argc) { total_samples = std::stoull(argv[++i]); }
        else if (a == "--bounds" && i+1 < argc) {
            std::string s = argv[++i];
            std::vector<double> mins, maxs;
            if (!parse_bounds_string(s, mins, maxs)) {
                std::cerr << "Error: could not parse --bounds value. Use formats like 'min1,min2;max1,max2' or 'min1:max1,min2:max2' or 'min:max'\n";
                return EXIT_FAILURE;
            }
            bounds_min = mins;
            bounds_max = maxs;
        }
        else if (a == "--bounds-min" && i+1 < argc) { bounds_min = parse_comma_doubles(argv[++i]); }
        else if (a == "--bounds-max" && i+1 < argc) { bounds_max = parse_comma_doubles(argv[++i]); }
        else if (a == "--dims" && i+1 < argc) { explicit_dims = std::stoi(argv[++i]); }
        else if (a == "--tolerance" && i+1 < argc) { tolerance = std::stod(argv[++i]); }
        else if (a == "--half" || a == "--fp16") { enable_fp16 = true; }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: ./mci_optimized --func \"FUNC\" --bounds \"min1:max1,min2:max2\" --sample N [--dims D] [--tolerance T] [--half]\n";
            std::cout << "Alternate (backwards compatible): --expr, --samples, --bounds-min, --bounds-max\n";
            std::cout << "Options:\n";
            std::cout << "  --half, --fp16   Enable FP16 (half precision) support\n";
            return EXIT_SUCCESS;
        }
    }

    // Auto-detect variables/dimensions from expression unless explicitly provided
    auto vars = extract_variables(expr);
    int dims = explicit_dims > 0 ? explicit_dims : static_cast<int>(vars.size());
    if (dims <= 0) dims = 1;

    if (bounds_min.empty()) bounds_min.assign(dims, 0.0);
    if (bounds_max.empty()) bounds_max.assign(dims, 1.0);

    if (bounds_min.size() == 1 && dims > 1) bounds_min.assign(dims, bounds_min[0]);
    if (bounds_max.size() == 1 && dims > 1) bounds_max.assign(dims, bounds_max[0]);

    if ((int)bounds_min.size() != dims || (int)bounds_max.size() != dims) {
        std::cerr << "Error: bounds size does not match detected/explicit dims (" << dims << ").\n";
        return EXIT_FAILURE;
    }

    std::cout << "Using expression: " << expr << "\n";
    std::cout << "Detected variables: ";
    if (vars.empty()) {
        std::cout << "(none - constant expression)";
    } else {
        for (size_t i = 0; i < vars.size(); ++i) {
            std::cout << vars[i];
            if (i < vars.size() - 1) std::cout << ", ";
        }
    }
    std::cout << "\n";
    std::cout << "Detected dimensions: " << dims << " (max 10)\n";
    std::cout << "Total samples: " << total_samples << "\n";
    std::cout << "Error tolerance: " << tolerance << "\n";
    
    // Compile expression
    auto terms = split_expression(expr);
    std::vector<CompiledExpr> compiled_exprs;
    for (const auto& term : terms) {
        auto tokens = tokenize(term);
        auto postfix = to_postfix(tokens);
        compiled_exprs.push_back(compile_expression(postfix, dims, &vars));
    }
    
    std::cout << "\n=== Expression Analysis ===\n";
    std::cout << "Number of terms: " << terms.size() << "\n";
    double volume = 1.0;
    for (int d = 0; d < dims; ++d) volume *= (bounds_max[d] - bounds_min[d]);
    std::cout << "Integration domain volume: " << std::scientific 
              << std::setprecision(4) << volume << "\n";
    
    // Warmup
    if (enable_fp16) {
        monte_carlo_integrate_nd_cuda_batch_fp16(
            1000, bounds_min, bounds_max, compiled_exprs, config);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // FIXED: Pass all terms for accuracy analysis
    analyze_accuracy_comprehensive(terms, bounds_min, bounds_max, compiled_exprs, config, enable_fp16);
    
    // ========================================================================
    // PRECISION-SPECIFIC BENCHMARKS
    // ========================================================================
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    double time_fp16 = 0, time_fp32 = 0, time_fp64 = 0;

    // FP16 Benchmark
    if (enable_fp16) {
        std::cout << "\n=== FP16 (Half Precision) Benchmark ===\n";
        {
            CUDA_CHECK(cudaEventRecord(start, 0));
            auto results = monte_carlo_integrate_nd_cuda_batch_fp16(
                total_samples, bounds_min, bounds_max, compiled_exprs, config);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaDeviceSynchronize());

            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            time_fp16 = ms / 1000.0;

            double total = 0.0;
            for (size_t i = 0; i < results.size(); ++i) {
                std::cout << "  Term \"" << terms[i] << "\" = "
                          << std::fixed << std::setprecision(8) << results[i] << "\n";
                total += results[i];
            }

            std::cout << "\nTotal result: " << std::setprecision(10) << total << "\n";
            auto metrics = calculate_metrics(ms, total_samples, 15);
            metrics.print();
        }
    }
    
    // FP32 Benchmark
    std::cout << "\n=== FP32 (Single Precision) Benchmark ===\n";
    {
        CUDA_CHECK(cudaEventRecord(start, 0));
        auto results = monte_carlo_integrate_nd_cuda_batch<float>(
            total_samples, bounds_min, bounds_max, compiled_exprs, config);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        time_fp32 = ms / 1000.0;
        
        double total = 0.0;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  Term \"" << terms[i] << "\" = " 
                      << std::fixed << std::setprecision(10) << results[i] << "\n";
            total += results[i];
        }
        
        std::cout << "\nTotal result: " << std::setprecision(10) << total << "\n";
        auto metrics = calculate_metrics(ms, total_samples, 15);
        metrics.print();
    }
    
    // FP64 Benchmark
    std::cout << "\n=== FP64 (Double Precision) Benchmark ===\n";
    {
        CUDA_CHECK(cudaEventRecord(start, 0));
        auto results = monte_carlo_integrate_nd_cuda_batch<double>(
            total_samples, bounds_min, bounds_max, compiled_exprs, config);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        time_fp64 = ms / 1000.0;
        
        double total = 0.0;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "  Term \"" << terms[i] << "\" = " 
                      << std::fixed << std::setprecision(14) << results[i] << "\n";
            total += results[i];
        }
        
        std::cout << "\nTotal result: " << std::setprecision(14) << total << "\n";
        auto metrics = calculate_metrics(ms, total_samples, 15);
        metrics.print();
    }
    
    compare_speedups(time_fp16, time_fp32, time_fp64);
    
    // ========================================================================
    // TERM-WISE ADAPTIVE MIXED PRECISION
    // ========================================================================
    std::cout << "\n=== TERM-WISE ADAPTIVE MIXED PRECISION ===\n";
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<Precision> term_precisions(terms.size());

        #pragma omp parallel for
        for (size_t i = 0; i < terms.size(); ++i) {
            auto tokens = tokenize(terms[i]);
            auto postfix = to_postfix(tokens);
            term_precisions[i] = select_precision_for_term(
                postfix, bounds_min, bounds_max, tolerance, terms[i], &vars);
        }

        // If FP16 is disabled, upgrade Half precision to Float
        if (!enable_fp16) {
            for (auto& p : term_precisions) {
                if (p == Precision::Half) p = Precision::Float;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double selection_time = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "Precision selection: " << std::fixed << std::setprecision(4)
                  << selection_time << " s\n";

        size_t cnt_h = 0, cnt_f = 0, cnt_d = 0;
        for (auto p : term_precisions) {
            if (p == Precision::Half) ++cnt_h;
            else if (p == Precision::Float) ++cnt_f;
            else ++cnt_d;
        }
        std::cout << "Classification:\n";
        std::cout << "  Half→FP16: " << cnt_h << " terms\n";
        std::cout << "  Float→FP32: " << cnt_f << " terms\n";
        std::cout << "  Double→FP64: " << cnt_d << " terms\n\n";

        int fast_samples = (dims <= 4) ? 96 : (dims <= 7) ? 48 : 24;

        std::vector<long double> variances(terms.size(), 0.0L);
        #pragma omp parallel for
        for (size_t i = 0; i < terms.size(); ++i) {
            auto tokens = tokenize(terms[i]);
            auto postfix = to_postfix(tokens);
            auto m = analyze_expression_fast(postfix, bounds_min, bounds_max, fast_samples, &vars);
            variances[i] = std::max(0.0L, m.var);
        }

        auto cost_of = [](Precision p){ 
            if (p==Precision::Half) return 0.5L; 
            if (p==Precision::Float) return 1.0L; 
            return 8.0L; 
        };

        std::vector<size_t> samples_per_term(terms.size(), 1);
        long double total_weight = 0.0L;
        std::vector<long double> weights(terms.size(), 0.0L);

        size_t min_samples_per_term = total_samples / (terms.size() * 10);
        min_samples_per_term = std::max(min_samples_per_term, (size_t)100000);

        for (size_t i = 0; i < terms.size(); ++i) {
            long double v = std::sqrt(variances[i]);  // Use stddev instead of variance
            if (v <= 0) v = 1e-12L;
            long double w = v / cost_of(term_precisions[i]);
            weights[i] = w;
            total_weight += w;
        }

        if (total_weight <= 0) {
            for (size_t i = 0; i < terms.size(); ++i)
                samples_per_term[i] = total_samples / terms.size();
        } else {
            size_t assigned = 0;
            for (size_t i = 0; i < terms.size(); ++i) {
                size_t n = std::max(min_samples_per_term,
                    (size_t)std::llround((long double)total_samples * (weights[i] / total_weight)));
                samples_per_term[i] = n;
                assigned += n;
            }
            if (assigned != total_samples) {
                long diff = (long)total_samples - (long)assigned;
                std::vector<size_t> idx(terms.size());
                for (size_t i=0; i<terms.size(); ++i) idx[i]=i;
                std::sort(idx.begin(), idx.end(), 
                    [&](size_t a, size_t b){ return weights[a] > weights[b]; });
                size_t k = 0;
                while (diff != 0) {
                    if (diff > 0) { 
                        samples_per_term[idx[k%idx.size()]]++; 
                        diff--; 
                    } else { 
                        if (samples_per_term[idx[k%idx.size()]]>1) { 
                            samples_per_term[idx[k%idx.size()]]--; 
                            diff++; 
                        } 
                    }
                    k++;
                }
            }
        }

        std::cout << "\nSample allocation per term:\n";
        for (size_t i = 0; i < terms.size(); ++i) {
            std::cout << "  Term " << i << " [" << (term_precisions[i] == Precision::Double ? "FP64" :
                         term_precisions[i] == Precision::Float ? "FP32" : "FP16")
                      << "]: " << samples_per_term[i] << " samples ("
                      << std::setprecision(2) << (100.0 * samples_per_term[i] / total_samples) << "%)\n";
        }
        std::cout << "\n";

        CUDA_CHECK(cudaEventRecord(start, 0));

        auto mixed_res = monte_carlo_integrate_nd_cuda_mixed(
            bounds_min, bounds_max, compiled_exprs,
            term_precisions, samples_per_term, config);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        double total_mixed = 0.0;
        for (size_t i = 0; i < mixed_res.size(); ++i) {
            auto p = term_precisions[i];
            if (p == Precision::Half) {
                std::cout << "[FP16] \"" << terms[i] << "\" = " << std::fixed 
                          << std::setprecision(8) << mixed_res[i] << "\n";
            } else if (p == Precision::Float) {
                std::cout << "[FP32] \"" << terms[i] << "\" = " << std::fixed 
                          << std::setprecision(10) << mixed_res[i] << "\n";
            } else {
                std::cout << "[FP64] \"" << terms[i] << "\" = " << std::fixed 
                          << std::setprecision(14) << mixed_res[i] << "\n";
            }
            total_mixed += mixed_res[i];
        }

        float ms_kernel;
        CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, start, stop));
        double total_time = selection_time + (ms_kernel / 1000.0);
        
        std::cout << "\nMixed precision result: " << std::setprecision(10) << total_mixed << "\n";
        std::cout << "Total time: " << std::setprecision(4) << total_time << " s\n";
        std::cout << "Selection overhead: " << std::setprecision(2) 
                  << (selection_time/total_time*100) << "%\n";
        
        double avg_cost = (cnt_h * 0.5 + cnt_f * 1.0 + cnt_d * 8.0) / terms.size();
        std::cout << "Avg computational cost: " << std::setprecision(2) 
                  << avg_cost << "x (FP32=1x baseline)\n";
        std::cout << "Savings vs pure FP64: " << std::setprecision(1) 
                  << ((8.0-avg_cost)/8.0*100) << "%\n";
    }
    
    // ========================================================================
    // ADAPTIVE REGION-WISE MIXED PRECISION
    // ========================================================================
    std::cout << "\n=== ADAPTIVE REGION-WISE MIXED PRECISION ===\n";
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        auto postfix = to_postfix(tokenize(expr));
        Region initial(bounds_min, bounds_max);
        size_t max_regions = 256;
        auto regions = adaptive_partition_nd(postfix, initial, 0.001, 1.0, max_regions, &vars);

        std::cout << "Created " << regions.size() << " adaptive regions\n";
        std::vector<Precision> region_precisions(regions.size());
        #pragma omp parallel for
        for (size_t i = 0; i < regions.size(); ++i) {
            region_precisions[i] = select_precision_for_region(
                postfix, regions[i], tolerance, "Region " + std::to_string(i), &vars);
        }

        // If FP16 is disabled, upgrade Half precision to Float
        if (!enable_fp16) {
            for (auto& p : region_precisions) {
                if (p == Precision::Half) p = Precision::Float;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double selection_time = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "Precision selection: " << std::fixed << std::setprecision(4)
                  << selection_time << " s\n";

        size_t r_h = 0, r_f = 0, r_d = 0;
        for (auto p : region_precisions) {
            if (p==Precision::Half) ++r_h;
            else if (p==Precision::Float) ++r_f;
            else ++r_d;
        }
        std::cout << "Classification:\n";
        std::cout << "  Half→FP16: " << r_h << " regions\n";
        std::cout << "  Float→FP32: " << r_f << " regions\n";
        std::cout << "  Double→FP64: " << r_d << " regions\n\n";

        int region_samples = (dims <= 4) ? 48 : (dims <= 7) ? 24 : 12;

        CompiledExpr compiled_full = compile_expression(postfix, dims, &vars);
        std::vector<long double> region_vars(regions.size(), 0.0L);
        #pragma omp parallel for
        for (size_t i = 0; i < regions.size(); ++i) {
            auto m = analyze_expression_fast(postfix, regions[i].bounds_min,
                                            regions[i].bounds_max, region_samples, &vars);
            region_vars[i] = std::max(0.0L, m.var);
        }

        std::vector<size_t> samples_per_region_vec(regions.size(), 1);
        long double total_w_reg = 0.0L;
        std::vector<long double> wreg(regions.size(), 0.0L);

        size_t min_samples_per_region = total_samples / (regions.size() * 10);
        min_samples_per_region = std::max(min_samples_per_region, (size_t)10000);

        for (size_t i = 0; i < regions.size(); ++i) {
            long double v = std::sqrt(region_vars[i]);  // Use stddev instead of variance
            if (v <= 0) v = 1e-12L;
            long double cost = (region_precisions[i]==Precision::Half ? 0.5L :
                               (region_precisions[i]==Precision::Float ? 1.0L : 8.0L));
            long double w = v / cost;
            wreg[i] = w;
            total_w_reg += w;
        }

        if (total_w_reg <= 0) {
            for (size_t i = 0; i < regions.size(); ++i)
                samples_per_region_vec[i] = total_samples / regions.size();
        } else {
            size_t assigned = 0;
            for (size_t i = 0; i < regions.size(); ++i) {
                size_t n = std::max(min_samples_per_region,
                    (size_t)std::llround((long double)total_samples * (wreg[i] / total_w_reg)));
                samples_per_region_vec[i] = n;
                assigned += n;
            }
            
            if (assigned != total_samples) {
                long diff = (long)total_samples - (long)assigned;
                std::vector<size_t> idx(regions.size()); 
                for (size_t i=0; i<regions.size(); ++i) idx[i]=i;
                std::sort(idx.begin(), idx.end(), 
                    [&](size_t a, size_t b){ return wreg[a] > wreg[b]; });
                size_t k=0; 
                while (diff != 0) { 
                    if (diff>0) { 
                        samples_per_region_vec[idx[k%idx.size()]]++; 
                        diff--; 
                    } else { 
                        if (samples_per_region_vec[idx[k%idx.size()]]>1) { 
                            samples_per_region_vec[idx[k%idx.size()]]--; 
                            diff++; 
                        } 
                    } 
                    k++; 
                }
            }
        }
        // Create vector with same expression for all regions
        std::vector<CompiledExpr> region_exprs_vec(regions.size(), compiled_full);

        // Flatten all region bounds into vectors for batch processing
        std::vector<double> all_bounds_min, all_bounds_max;
        for (const auto& region : regions) {
            all_bounds_min.insert(all_bounds_min.end(), region.bounds_min.begin(), region.bounds_min.end());
            all_bounds_max.insert(all_bounds_max.end(), region.bounds_max.begin(), region.bounds_max.end());
        }

        CUDA_CHECK(cudaEventRecord(start, 0));

        // Process all regions in ONE batch call using the mixed precision kernel
        // Build per-region bounds vectors
        std::vector<std::vector<double>> bounds_min_per_region, bounds_max_per_region;
        for (const auto& r : regions) {
            bounds_min_per_region.push_back(r.bounds_min);
            bounds_max_per_region.push_back(r.bounds_max);
        }

        auto region_results = monte_carlo_integrate_nd_cuda_batch_mixed(
            0,  // unused when samples_per_region_vec is provided
            bounds_min,  // global bounds (not used)
            bounds_max,
            bounds_min_per_region,
            bounds_max_per_region,
            samples_per_region_vec,
            region_exprs_vec,
            region_precisions,
            config
        );

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        double total_region = 0.0;
        for (double r : region_results) {
            total_region += r;
        }
        float ms_kernel;
        CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, start, stop));
        double total_time = selection_time + (ms_kernel / 1000.0);
        
        std::cout << "\nRegion-wise result: " << std::setprecision(10) << total_region << "\n";
        std::cout << "Total time: " << std::setprecision(4) << total_time << " s\n";
        std::cout << "Selection overhead: " << std::setprecision(2) 
                  << (selection_time/total_time*100) << "%\n";
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
