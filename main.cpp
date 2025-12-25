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
    const GPUConfig& config)
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
    
    std::cout << std::setw(12) << "Samples" 
              << std::setw(18) << "FP16 Rel.Error"
              << std::setw(18) << "FP32 Rel.Error"
              << std::setw(18) << "FP64 Rel.Error"
              << std::setw(18) << "FP16 Value"
              << std::setw(18) << "FP32 Value"
              << std::setw(18) << "FP64 Value" << "\n";
    std::cout << std::string(120, '-') << "\n";
    
    struct Result { 
        size_t samples; 
        double fp16_err, fp32_err, fp64_err; 
        double fp16_val, fp32_val, fp64_val; 
    };
    std::vector<Result> results;
    
    for (size_t n : sample_counts) {
        Result r;
        r.samples = n;
        
        // FP16 - sum all terms
        auto res_fp16 = monte_carlo_integrate_nd_cuda_batch_fp16(
            n, bounds_min, bounds_max, compiled, config);
        r.fp16_val = 0.0;
        for (auto v : res_fp16) r.fp16_val += v;
        r.fp16_err = std::abs(r.fp16_val - analytical) / std::abs(analytical);
        
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
        
        std::cout << std::setw(12) << n
                  << std::setw(18) << std::scientific << std::setprecision(4) << r.fp16_err
                  << std::setw(18) << r.fp32_err
                  << std::setw(18) << r.fp64_err
                  << std::setw(18) << std::fixed << std::setprecision(8) << r.fp16_val
                  << std::setw(18) << r.fp32_val
                  << std::setw(18) << std::setprecision(10) << r.fp64_val << "\n";
        
        results.push_back(r);
    }
    
    // Convergence analysis
    std::cout << "\n=== Error Convergence Analysis ===\n";
    std::cout << "MC error should scale as O(1/sqrt(N))\n";
    
    if (results.size() >= 2) {
        double n1 = results[results.size()-2].samples;
        double n2 = results[results.size()-1].samples;
        double theoretical = std::sqrt(n2 / n1);
        
        double fp16_imp = results[results.size()-2].fp16_err / 
                         std::max(1e-15, results[results.size()-1].fp16_err);
        double fp32_imp = results[results.size()-2].fp32_err / 
                         std::max(1e-15, results[results.size()-1].fp32_err);
        double fp64_imp = results[results.size()-2].fp64_err / 
                         std::max(1e-15, results[results.size()-1].fp64_err);
        
        std::cout << "Theoretical improvement (" << (int)n1 << " -> " << (int)n2 << "): " 
                  << std::fixed << std::setprecision(2) << theoretical << "x\n";
        std::cout << "Actual improvements:\n";
        std::cout << "  FP16: " << fp16_imp << "x\n";
        std::cout << "  FP32: " << fp32_imp << "x\n";
        std::cout << "  FP64: " << fp64_imp << "x\n";
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

static std::vector<std::string> extract_variables(const std::string &expr) {
    static const std::set<std::string> funcs = {
        "sin","cos","tan","asin","acos","atan",
        "sinh","cosh","tanh","log","log10","exp",
        "sqrt","pow","abs","min","max"
    };
    std::vector<std::string> vars;
    std::set<std::string> seen;
    std::regex re("([A-Za-z_]\\w*)");
    for (auto it = std::sregex_iterator(expr.begin(), expr.end(), re); it != std::sregex_iterator(); ++it) {
        std::string tok = (*it)[1].str();
        if (funcs.count(tok)) continue;
        if (tok == "pi" || tok == "e") continue;
        if (seen.insert(tok).second) vars.push_back(tok);
    }
    return vars;
}

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
    GPUConfig config = get_optimal_gpu_config();
    
    // Defaults (used if not overridden via CLI)
    std::string expr = "sin(x + y + z + w)+ cos(x*y) - log(1 + z*w) * x^5 + y^4 * exp(-w) - z^2 + x^12 - w^3 + sin(z*w^2) - 4";
    size_t total_samples = 100000000; // 100M
    double tolerance = 1e-5;
    std::vector<double> bounds_min;
    std::vector<double> bounds_max;
    int explicit_dims = -1;

    // Simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--expr" && i+1 < argc) { expr = argv[++i]; }
        else if (a == "--samples" && i+1 < argc) { total_samples = std::stoull(argv[++i]); }
        else if (a == "--bounds-min" && i+1 < argc) { bounds_min = parse_comma_doubles(argv[++i]); }
        else if (a == "--bounds-max" && i+1 < argc) { bounds_max = parse_comma_doubles(argv[++i]); }
        else if (a == "--dims" && i+1 < argc) { explicit_dims = std::stoi(argv[++i]); }
        else if (a == "--tolerance" && i+1 < argc) { tolerance = std::stod(argv[++i]); }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: ./mci [--expr \"EXPR\"] [--samples N] [--bounds-min a,b,...] [--bounds-max a,b,...] [--dims D] [--tolerance T]\n";
            return EXIT_SUCCESS;
        }
    }

    // Auto-detect variables/dimensions from expression unless explicitly provided
    auto vars = extract_variables(expr);
    int dims = explicit_dims > 0 ? explicit_dims : static_cast<int>(vars.size());
    if (dims <= 0) dims = 1;

    if (bounds_min.empty()) bounds_min.assign(dims, 0.0);
    if (bounds_max.empty()) bounds_max.assign(dims, 1.0);

    if ((int)bounds_min.size() != dims || (int)bounds_max.size() != dims) {
        std::cerr << "Error: bounds size does not match detected/explicit dims (" << dims << ").\n";
        return EXIT_FAILURE;
    }

    std::cout << "Using expression: " << expr << "\n";
    std::cout << "Detected dims: " << dims << "\n";
    std::cout << "Total samples: " << total_samples << "\n";
    std::cout << "Error tolerance: " << tolerance << " (default)\n";
    
    std::cout << "Error tolerance: " << tolerance << " (default)\n";
    
    // Compile expression
    auto terms = split_expression(expr);
    std::vector<CompiledExpr> compiled_exprs;
    for (const auto& term : terms) {
        auto tokens = tokenize(term);
        auto postfix = to_postfix(tokens);
        compiled_exprs.push_back(compile_expression(postfix, dims));
    }
    
    std::cout << "\n=== Expression Analysis ===\n";
    std::cout << "Number of terms: " << terms.size() << "\n";
    double volume = 1.0;
    for (int d = 0; d < dims; ++d) volume *= (bounds_max[d] - bounds_min[d]);
    std::cout << "Integration domain volume: " << std::scientific 
              << std::setprecision(4) << volume << "\n";
    
    // Warmup
    monte_carlo_integrate_nd_cuda_batch_fp16(
        1000, bounds_min, bounds_max, compiled_exprs, config);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // FIXED: Pass all terms for accuracy analysis
    analyze_accuracy_comprehensive(terms, bounds_min, bounds_max, compiled_exprs, config);
    
    // ========================================================================
    // PRECISION-SPECIFIC BENCHMARKS
    // ========================================================================
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    double time_fp16 = 0, time_fp32 = 0, time_fp64 = 0;
    
    // FP16 Benchmark
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
                postfix, bounds_min, bounds_max, tolerance, terms[i]);
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

        CUDA_CHECK(cudaEventRecord(start, 0));

        // Variance-based sample allocation
        std::vector<long double> variances(terms.size(), 0.0L);
        #pragma omp parallel for
        for (size_t i = 0; i < terms.size(); ++i) {
            auto tokens = tokenize(terms[i]);
            auto postfix = to_postfix(tokens);
            auto m = analyze_expression_fast(postfix, bounds_min, bounds_max, 64);
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
        for (size_t i = 0; i < terms.size(); ++i) {
            long double v = variances[i];
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
                size_t n = std::max((size_t)1, 
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

        std::vector<std::vector<double>> bmin_per_term(terms.size(), bounds_min);
        std::vector<std::vector<double>> bmax_per_term(terms.size(), bounds_max);

        auto mixed_res = monte_carlo_integrate_nd_cuda_batch_mixed(
            total_samples, bounds_min, bounds_max, bmin_per_term, bmax_per_term,
            samples_per_term, compiled_exprs, term_precisions, config);

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
        
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
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
        
        // Create adaptive regions
        auto postfix = to_postfix(tokenize(expr));
        Region initial(bounds_min, bounds_max);
        auto regions = adaptive_partition_nd(postfix, initial, 1e-2, 1e2, 256);
        
        std::cout << "Created " << regions.size() << " adaptive regions\n";
        
        // Precision selection per region
        std::vector<Precision> region_precisions(regions.size());
        #pragma omp parallel for
        for (size_t i = 0; i < regions.size(); ++i) {
            region_precisions[i] = select_precision_for_region(
                postfix, regions[i], tolerance, "Region " + std::to_string(i));
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
        
        CompiledExpr compiled_full = compile_expression(postfix, dims);

        // Variance-based sample allocation for regions
        std::vector<long double> region_vars(regions.size(), 0.0L);
        #pragma omp parallel for
        for (size_t i = 0; i < regions.size(); ++i) {
            auto m = analyze_expression_fast(postfix, regions[i].bounds_min, 
                                            regions[i].bounds_max, 64);
            region_vars[i] = std::max(0.0L, m.var);
        }

        std::vector<size_t> samples_per_region_vec(regions.size(), 1);
        long double total_w_reg = 0.0L;
        std::vector<long double> wreg(regions.size(), 0.0L);
        
        for (size_t i = 0; i < regions.size(); ++i) {
            long double v = region_vars[i]; 
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
                size_t n = std::max((size_t)1, 
                    (size_t)std::llround((long double)total_samples * (wreg[i] / total_w_reg)));
                samples_per_region_vec[i] = n; 
                assigned += n;
            }
            
            // Adjust to match total_samples exactly
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

        // Prepare region data
        std::vector<CompiledExpr> compiled_regions(regions.size(), compiled_full);
        std::vector<std::vector<double>> bmin_regions(regions.size()), 
                                         bmax_regions(regions.size());
        for (size_t i = 0; i < regions.size(); ++i) { 
            bmin_regions[i] = regions[i].bounds_min; 
            bmax_regions[i] = regions[i].bounds_max; 
        }

        // Execute region-wise integration
        CUDA_CHECK(cudaEventRecord(start, 0));
        
        auto region_res = monte_carlo_integrate_nd_cuda_batch_mixed(
            total_samples, bounds_min, bounds_max, bmin_regions, bmax_regions,
            samples_per_region_vec, compiled_regions, region_precisions, config);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Print results
        for (size_t i = 0; i < region_res.size(); ++i) {
            auto p = region_precisions[i];
            if (p == Precision::Half) {
                std::cout << "[FP16] Region " << i << " = " << std::fixed 
                          << std::setprecision(8) << region_res[i] << "\n";
            } else if (p == Precision::Float) {
                std::cout << "[FP32] Region " << i << " = " << std::fixed 
                          << std::setprecision(10) << region_res[i] << "\n";
            } else {
                std::cout << "[FP64] Region " << i << " = " << std::fixed 
                          << std::setprecision(14) << region_res[i] << "\n";
            }
        }
        
        float ms_kernel;
        CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, start, stop));
        double total_time = selection_time + (ms_kernel / 1000.0);
        
        // Sum all region results
        double total_region = 0.0;
        for (double r : region_res) total_region += r;
        
        std::cout << "\nRegion-wise result: " << std::setprecision(10) << total_region << "\n";
        std::cout << "Total time: " << std::setprecision(4) << total_time << " s\n";
        std::cout << "Selection overhead: " << std::setprecision(2) 
                  << (selection_time/total_time*100) << "%\n";
        
        double avg_cost = (r_h * 0.5 + r_f * 1.0 + r_d * 8.0) / regions.size();
        std::cout << "Savings vs pure FP64: " << std::setprecision(1) 
                  << ((8.0-avg_cost)/8.0*100) << "%\n";
    }
    
    // ========================================================================
    // CLEANUP AND EXIT
    // ========================================================================
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDeviceReset());
    
    std::cout << "\n================================================================\n";
    std::cout << "  Analysis complete!\n";
    std::cout << "================================================================\n";
    
    return EXIT_SUCCESS;
}