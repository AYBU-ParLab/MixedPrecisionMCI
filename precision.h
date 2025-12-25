#ifndef PRECISION_H
#define PRECISION_H

#include <vector>
#include <string>
#include <stack>
#include <cmath>
#include <random>
#include <limits>
#include <iostream>
#include <omp.h>
#include <queue>
#include "parser.h"

const int FIXED_SEED = 42;

enum class Precision {
    Half,           // FP16 - 16-bit
    Float,          // FP32
    Double,         // FP64
    LongDouble      // FP80/128
};

// Multi-dimensional region structure with adaptive partitioning support
struct Region {
    std::vector<double> bounds_min;
    std::vector<double> bounds_max;
    double error_estimate;
    int refinement_level;
    
    Region() : error_estimate(0.0), refinement_level(0) {}
    
    Region(const std::vector<double>& min_b, const std::vector<double>& max_b) 
        : bounds_min(min_b), bounds_max(max_b), error_estimate(0.0), refinement_level(0) {}
    
    Region(double x1, double x2, double y1, double y2) 
        : error_estimate(0.0), refinement_level(0) {
        bounds_min = {x1, y1};
        bounds_max = {x2, y2};
    }
    
    double volume() const {
        double vol = 1.0;
        for (size_t i = 0; i < bounds_min.size(); ++i) {
            vol *= (bounds_max[i] - bounds_min[i]);
        }
        return vol;
    }
    
    std::vector<Region> subdivide() const {
        std::vector<Region> children;
        int dims = bounds_min.size();
        
        int split_dim = 0;
        double max_span = bounds_max[0] - bounds_min[0];
        for (int i = 1; i < dims; ++i) {
            double span = bounds_max[i] - bounds_min[i];
            if (span > max_span) {
                max_span = span;
                split_dim = i;
            }
        }
        
        double mid = 0.5 * (bounds_min[split_dim] + bounds_max[split_dim]);
        
        for (int i = 0; i < 2; ++i) {
            Region child;
            child.bounds_min = bounds_min;
            child.bounds_max = bounds_max;
            child.refinement_level = refinement_level + 1;
            
            if (i == 0) {
                child.bounds_max[split_dim] = mid;
            } else {
                child.bounds_min[split_dim] = mid;
            }
            
            children.push_back(child);
        }
        
        return children;
    }
};

// ======================
// FAST Expression Evaluation
// ======================
template<typename T>
inline T evaluate_postfix_fast(const std::vector<Token>& postfix, const T* vars, int dims) {
    T stack[64];
    int sp = 0;
    
    for (const auto& token : postfix) {
        if (token.type == TokenType::Number) {
            stack[sp++] = static_cast<T>(std::stold(token.value));
        } else if (token.type == TokenType::Variable) {
            int idx = -1;
            if (token.value == "x") idx = 0;
            else if (token.value == "y") idx = 1;
            else if (token.value == "z") idx = 2;
            else if (token.value == "w") idx = 3;
            
            if (idx >= 0 && idx < dims) {
                stack[sp++] = vars[idx];
            } else {
                stack[sp++] = 0;
            }
        } else if (token.type == TokenType::Operator) {
            T b = stack[--sp];
            T a = stack[--sp];
            if (token.value == "+") stack[sp++] = a + b;
            else if (token.value == "-") stack[sp++] = a - b;
            else if (token.value == "*") stack[sp++] = a * b;
            else if (token.value == "/") stack[sp++] = (b != 0) ? (a / b) : 0;
            else if (token.value == "^") stack[sp++] = std::pow(a, b);
        } else if (token.type == TokenType::Function) {
            T a = stack[--sp];
            if (token.value == "sin") stack[sp++] = std::sin(a);
            else if (token.value == "cos") stack[sp++] = std::cos(a);
            else if (token.value == "log") stack[sp++] = (a > 0) ? std::log10(a) : 0;
            else if (token.value == "ln") stack[sp++] = (a > 0) ? std::log(a) : 0;
            else if (token.value == "exp") stack[sp++] = std::exp(a);
            else if (token.value == "sqrt") stack[sp++] = (a >= 0) ? std::sqrt(a) : 0;
            else if (token.value == "tan") stack[sp++] = std::tan(a);
            else if (token.value == "abs") stack[sp++] = std::abs(a);
        }
    }
    return (sp > 0) ? stack[0] : 0;
}

// ======================
// Statistical Metrics - ENHANCED FOR REGIONS
// ======================
struct StatisticalMetrics {
    long double avg;
    long double var;
    long double grad;
    long double max_val;
    long double min_val;
    long double range;
    long double coefficient_of_variation;
    int samples_used;
};

inline StatisticalMetrics analyze_expression_fast(
    const std::vector<Token>& postfix,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    int target_samples = 64)  // Increased for better region analysis
{
    StatisticalMetrics metrics = {0, 0, 0, -1e100L, 1e100L, 0, 0, 0};
    int dims = bounds_min.size();
    
    long double sum = 0, sum_sq = 0;
    long double max_grad = 0;
    long double max_val = -1e100L, min_val = 1e100L;
    int valid = 0;
    
    #pragma omp parallel default(none) \
        shared(postfix, bounds_min, bounds_max, dims, target_samples) \
        reduction(+:sum, sum_sq, valid) reduction(max:max_grad, max_val) reduction(min:min_val)
    {
        std::mt19937 gen(FIXED_SEED + omp_get_thread_num());
        std::vector<std::uniform_real_distribution<long double>> dists;
        for (int d = 0; d < dims; ++d) {
            dists.emplace_back(bounds_min[d], bounds_max[d]);
        }
        
        #pragma omp for
        for (int i = 0; i < target_samples; ++i) {
            long double vars[4] = {0, 0, 0, 0};
            for (int d = 0; d < dims; ++d) {
                vars[d] = dists[d](gen);
                if (vars[d] <= 1e-12L) vars[d] = 1e-12L;
            }
            
            try {
                long double val = evaluate_postfix_fast<long double>(postfix, vars, dims);
                sum += val;  // Use signed value for avg
                sum_sq += val * val;
                max_val = std::max(max_val, val);
                min_val = std::min(min_val, val);
                valid++;
                
                // Gradient estimation - all dimensions
                for (int d = 0; d < std::min(dims, 2); ++d) {  // First 2 dims for speed
                    long double h = (bounds_max[d] - bounds_min[d]) * 0.01;
                    long double vars_h[4];
                    for (int k = 0; k < dims; ++k) vars_h[k] = vars[k];
                    vars_h[d] += h;
                    if (vars_h[d] <= bounds_max[d]) {
                        long double val_h = evaluate_postfix_fast<long double>(postfix, vars_h, dims);
                        long double grad_est = std::abs((val_h - val) / h);
                        max_grad = std::max(max_grad, grad_est);
                    }
                }
            } catch (...) {}
        }
    }
    
    if (valid < 2) return metrics;
    
    metrics.avg = sum / valid;
    metrics.var = (sum_sq / valid) - (metrics.avg * metrics.avg);
    metrics.grad = max_grad;
    metrics.max_val = max_val;
    metrics.min_val = min_val;
    metrics.range = max_val - min_val;
    metrics.coefficient_of_variation = std::sqrt(std::abs(metrics.var)) / (std::abs(metrics.avg) + 1e-10L);
    metrics.samples_used = valid;
    
    return metrics;
}

// Single-threaded, lightweight variant for use inside parallel regions
inline StatisticalMetrics analyze_expression_fast_serial(
    const std::vector<Token>& postfix,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    int target_samples = 16)
{
    StatisticalMetrics metrics = {0, 0, 0, -1e100L, 1e100L, 0, 0, 0};
    int dims = bounds_min.size();

    long double sum = 0, sum_sq = 0;
    long double max_grad = 0;
    long double max_val = -1e100L, min_val = 1e100L;
    int valid = 0;

    std::mt19937 gen(FIXED_SEED);
    std::vector<std::uniform_real_distribution<long double>> dists;
    for (int d = 0; d < dims; ++d) {
        dists.emplace_back(bounds_min[d], bounds_max[d]);
    }

    for (int i = 0; i < target_samples; ++i) {
        std::vector<long double> vars(dims);
        for (int d = 0; d < dims; ++d) {
            vars[d] = dists[d](gen);
            if (vars[d] <= 1e-12L) vars[d] = 1e-12L;
        }

        try {
            long double val = evaluate_postfix_fast<long double>(postfix, vars.data(), dims);
            sum += val;
            sum_sq += val * val;
            max_val = std::max(max_val, val);
            min_val = std::min(min_val, val);
            valid++;

            // Simple gradient estimate (first two dims)
            for (int d = 0; d < std::min(dims, 2); ++d) {
                long double h = (bounds_max[d] - bounds_min[d]) * 0.01L;
                std::vector<long double> vars_h(vars.begin(), vars.end());
                vars_h[d] += h;
                if (vars_h[d] <= bounds_max[d]) {
                    long double val_h = evaluate_postfix_fast<long double>(postfix, vars_h.data(), dims);
                    long double grad_est = std::abs((val_h - val) / h);
                    if (grad_est > max_grad) max_grad = grad_est;
                }
            }
        } catch (...) {}
    }

    if (valid < 2) return metrics;

    metrics.avg = sum / valid;
    metrics.var = (sum_sq / valid) - (metrics.avg * metrics.avg);
    metrics.grad = max_grad;
    metrics.max_val = max_val;
    metrics.min_val = min_val;
    metrics.range = max_val - min_val;
    metrics.coefficient_of_variation = std::sqrt(std::abs(metrics.var)) / (std::abs(metrics.avg) + 1e-10L);
    metrics.samples_used = valid;

    return metrics;
}

// ======================
// GELIŞTIRILMIŞ PRECISION SELECTION - TERIM ÖZELLİKLERİNE GÖRE
// ======================
inline Precision select_precision_fast(
    const StatisticalMetrics& metrics,
    double tol,
    const std::string& name,
    bool termwise = false)
{
    // 1. CONSTANT CHECK - Sabit terimler -> FP16
    if (metrics.var == 0 && metrics.grad == 0) {
        std::cout << "  [" << name << "] Constant detected -> FP16\n";
        return Precision::Half;
    }
    
    // 2. BOUNDED SMOOTH FUNCTIONS - sin, cos gibi sınırlı fonksiyonlar
    // Avg ve max_val küçükse (< 2.0) ve varyans düşükse -> FP16
    if (termwise && metrics.max_val < 2.5L && metrics.avg < 2.0L) {
        long double normalized_var = metrics.var / (metrics.avg * metrics.avg + 1e-10L);
        if (normalized_var < 0.3L) {
            std::cout << "  [" << name << "] Bounded smooth function (max=" 
                      << metrics.max_val << ", var=" << normalized_var << ") -> FP16\n";
            return Precision::Half;
        }
    }
    
    // 3. SMALL MAGNITUDE - Küçük değerli terimler (|avg| < 1.0)
    if (termwise && std::abs(metrics.avg) < 1.0L && metrics.max_val < 1.5L) {
        long double grad_ratio = metrics.grad / (std::abs(metrics.avg) + 1e-10L);
        if (grad_ratio < 5.0L) {
            std::cout << "  [" << name << "] Small magnitude (avg=" 
                      << metrics.avg << ", max=" << metrics.max_val << ") -> FP16\n";
            return Precision::Half;
        }
    }
    
    // 4. POLYNOMIAL GROWTH CHECK - x^7, x^12 gibi büyük üslü terimler
    // Max değer çok büyükse (>100) veya gradient çok büyükse -> FP64
    if (termwise && (metrics.max_val > 100.0L || metrics.grad > 50.0L)) {
        std::cout << "  [" << name << "] Large growth detected (max=" 
                  << metrics.max_val << ", grad=" << metrics.grad << ") -> FP64\n";
        return Precision::Double;
    }
    
    // 5. LOGARITHMIC/PRECISION-SENSITIVE - log, hassas işlemler
    // Varyans/ortalama oranı yüksekse -> FP64
    if (termwise && metrics.var > 0) {
        long double cv = std::sqrt(std::abs(metrics.var)) / (std::abs(metrics.avg) + 1e-10L);
        if (cv > 1.5L && metrics.max_val > 5.0L) {
            std::cout << "  [" << name << "] High variance ratio (CV=" 
                      << cv << ") -> FP64\n";
            return Precision::Double;
        }
    }
    
    // 6. MEDIUM COMPLEXITY - Orta ölçekli terimler -> FP32
    if (termwise) {
        long double normalized_var = metrics.var / (metrics.avg * metrics.avg + 1e-10L);
        long double grad_ratio = metrics.grad / (std::abs(metrics.avg) + 1e-10L);
        
        // Orta düzeyde smooth ve orta büyüklük
        if (normalized_var < 0.5L && grad_ratio < 10.0L && metrics.max_val < 50.0L) {
            std::cout << "  [" << name << "] Medium complexity (var=" 
                      << normalized_var << ", grad_ratio=" << grad_ratio << ") -> FP32\n";
            return Precision::Float;
        }
    }
    
    const long double eps_h = 4.88e-4L;  // FP16: 2^-11
    const long double eps_f = std::numeric_limits<float>::epsilon();
    const long double eps_d = std::numeric_limits<double>::epsilon();
    
    long double max_val = std::max(std::abs(metrics.avg), std::sqrt(std::abs(metrics.var)));
    max_val = std::max(max_val, 1e-10L);
    
    long double norm_grad = std::max(metrics.grad, 1.0L);
    long double cond = norm_grad * max_val / std::max(std::abs(metrics.avg), static_cast<long double>(tol));
    cond = std::max(cond, 1.0L);
    cond = std::min(cond, 1000.0L);
    
    long double operation_count = termwise ? 5.0L : 8.0L;
    
    long double err_h = eps_h * max_val * cond * std::sqrt(operation_count);
    long double err_f = eps_f * max_val * cond * std::sqrt(operation_count);
    long double err_d = eps_d * max_val * cond * std::sqrt(operation_count);
    
    double safety_factor = termwise ? 5.0 : 1.0;  // Balanced safety

    if (err_h <= tol / safety_factor) {
        return Precision::Half;
    }
    if (err_f <= tol / safety_factor) {
        return Precision::Float;
    }
    if (err_d <= tol / safety_factor) {
        return Precision::Double;
    }
    
    std::cout << "  [" << name << "] High precision required -> FP64\n";
    return Precision::Double;
}

// ======================
// ADAPTIVE REGION PARTITIONING (priority-driven, bounded by max_regions)
// ======================
inline std::vector<Region> adaptive_partition_nd(
    const std::vector<Token>& postfix,
    const Region& initial_region,
    double variance_threshold = 1e-2,
    double gradient_threshold = 1e2,
    size_t max_regions = 256)
{
    struct PQItem { Region region; long double score; int depth; };
    auto cmp = [](const PQItem& a, const PQItem& b) { return a.score < b.score; };
    std::priority_queue<PQItem, std::vector<PQItem>, decltype(cmp)> pq(cmp);

    int dims = static_cast<int>(initial_region.bounds_min.size());

    std::cout << "\nAdaptive Partitioning (" << dims << "D)...\n";
    std::cout << "  Variance threshold: " << variance_threshold << "\n";
    std::cout << "  Gradient threshold: " << gradient_threshold << "\n";

    // Compute metric for initial region
    auto m0 = analyze_expression_fast(postfix, initial_region.bounds_min, initial_region.bounds_max, 50);
    long double score0 = std::max(m0.var / variance_threshold, m0.grad / gradient_threshold);
    pq.push(PQItem{initial_region, score0, 0});

    std::vector<Region> final_regions;

    // Continue splitting the most 'complex' region while we have budget and useful splits
    while (!pq.empty() && (final_regions.size() + pq.size()) < max_regions) {
        auto top = pq.top();
        if (top.score <= 1.0L) break; // no region exceeds thresholds
        pq.pop();

        Region current = top.region;

        // Don't split if too small
        double min_size = 1e-9; // tighter min size to avoid unnecessary early stopping
        bool too_small = false;
        for (int d = 0; d < dims; ++d) {
            if (current.bounds_max[d] - current.bounds_min[d] < min_size) { too_small = true; break; }
        }
        if (too_small) { final_regions.push_back(current); continue; }

        // Split by largest extent
        int split_dim = 0;
        double max_extent = current.bounds_max[0] - current.bounds_min[0];
        for (int d = 1; d < dims; ++d) {
            double extent = current.bounds_max[d] - current.bounds_min[d];
            if (extent > max_extent) { max_extent = extent; split_dim = d; }
        }

        double mid = 0.5 * (current.bounds_min[split_dim] + current.bounds_max[split_dim]);
        Region r1 = current; r1.bounds_max[split_dim] = mid;
        Region r2 = current; r2.bounds_min[split_dim] = mid;

        // Analyze children
        auto m1 = analyze_expression_fast(postfix, r1.bounds_min, r1.bounds_max, 40);
        auto m2 = analyze_expression_fast(postfix, r2.bounds_min, r2.bounds_max, 40);
        long double s1 = std::max(m1.var / variance_threshold, m1.grad / gradient_threshold);
        long double s2 = std::max(m2.var / variance_threshold, m2.grad / gradient_threshold);

        pq.push(PQItem{r1, s1, top.depth + 1});
        pq.push(PQItem{r2, s2, top.depth + 1});
    }

    // Flush remaining priority queue into final regions until max_regions
    while (!pq.empty() && final_regions.size() < max_regions) {
        final_regions.push_back(pq.top().region);
        pq.pop();
    }

    std::cout << "  Created " << final_regions.size() << " adaptive regions\n";
    return final_regions;
}

// ======================
// CONVENIENCE WRAPPERS
// ======================
inline Precision select_precision_for_term(
    const std::vector<Token>& postfix,
    const std::vector<double>& bounds_min,
    const std::vector<double>& bounds_max,
    double tol,
    const std::string& name)
{
    auto metrics = analyze_expression_fast(postfix, bounds_min, bounds_max, 64);
    return select_precision_fast(metrics, tol, name, true);
}

inline Precision select_precision_for_region(
    const std::vector<Token>& postfix,
    const Region& region,
    double tol,
    const std::string& name)
{
    // Two-stage cheap->heavy serial analysis to minimize selection overhead
    auto quick = analyze_expression_fast_serial(postfix, region.bounds_min, region.bounds_max, 6);

    // Fast rules based on cheap estimates
    if (std::abs(quick.avg) < 0.5 && quick.range < 0.5 && std::abs(quick.max_val) < 1.0 && quick.grad < 2.0) {
        return Precision::Half;
    }

    if (quick.max_val > 50.0 || quick.grad > 80.0) {
        return Precision::Double;
    }

    if (quick.coefficient_of_variation < 0.5 && quick.grad < 10.0 && std::abs(quick.avg) < 5.0) {
        return Precision::Float;
    }

    // Uncertain cases: run a slightly heavier serial analysis and use the enhanced selector
    auto metrics = analyze_expression_fast_serial(postfix, region.bounds_min, region.bounds_max, 32);
    return select_precision_fast(metrics, tol, name, false);
}

// ======================
// ENHANCED REGION-WISE PRECISION SELECTION (ported from newMixed_v2)
// ======================
inline Precision select_precision_enhanced(
    const StatisticalMetrics& metrics,
    double tol,
    const std::string& name,
    bool verbose = true)
{
    // Handle degenerate cases
    if (metrics.var == 0 && metrics.grad == 0) {
        if (verbose) {
            std::cout << "\n[" << name << "] Constant region detected -> FP16\n";
        }
        return Precision::Half;
    }

    // Quick classification for simple smooth regions
    if (metrics.grad > 0 && metrics.var > 0) {
        long double normalized_var = metrics.var / (metrics.avg * metrics.avg + 1e-10L);
        long double grad_ratio = metrics.grad / (std::abs(metrics.avg) + 1e-10L);

        if (normalized_var < 0.1 && grad_ratio < 1.5) {
            if (verbose) {
                std::cout << "\n[" << name << "] Simple smooth region detected -> FP16\n";
                std::cout << "  Normalized Var: " << normalized_var << ", Grad Ratio: " << grad_ratio << "\n";
            }
            return Precision::Half;
        }
    }

    const long double eps_half = 4.88e-4L; // FP16
    const long double eps_float = std::numeric_limits<float>::epsilon();
    const long double eps_double = std::numeric_limits<double>::epsilon();

    long double max_val = std::max(std::abs(metrics.avg), std::sqrt(std::abs(metrics.var)));
    max_val = std::max(max_val, 1e-10L);

    long double normalized_grad = std::max(metrics.grad, 1.0L);

    // Improved condition number with variance contribution
    long double condition_number = normalized_grad * max_val / std::max(std::abs(metrics.avg), static_cast<long double>(tol));
    condition_number += std::sqrt(metrics.var) / (std::abs(metrics.avg) + tol);
    condition_number = std::max(condition_number, 1.0L);

    // Adaptive operation count based on gradient complexity
    long double operation_count = 10.0L + std::log10(condition_number + 1);

    long double error_factor = max_val * condition_number * std::sqrt(operation_count);

    long double error_half = eps_half * error_factor;
    long double error_float = eps_float * error_factor;
    long double error_double = eps_double * error_factor;

    // Monte Carlo contribution with variance scaling
    long double mc_error_scale = std::sqrt(metrics.var) / (std::abs(metrics.avg) + 1e-10L);

    long double total_error_half = error_half * (1.0L + mc_error_scale);
    long double total_error_float = error_float * (1.0L + mc_error_scale);
    long double total_error_double = error_double * (1.0L + mc_error_scale);

    if (verbose) {
        std::cout << "\n=== Region Precision Analysis: " << name << " ===\n";
        std::cout << "  Avg: " << metrics.avg << ", Var: " << metrics.var << ", Grad: " << metrics.grad << "\n";
        std::cout << "  Condition: " << condition_number << ", Op Count: " << operation_count << "\n";
        std::cout << "  Half error:  " << total_error_half << "\n";
        std::cout << "  Float error: " << total_error_float << "\n";
        std::cout << "  Double:   " << total_error_double << "\n";
        std::cout << "  Tolerance:    " << tol << "\n";
    }

    double safety_factor = 5.0;

    if (total_error_half <= tol / safety_factor) {
        if (verbose) std::cout << "  -> Selected: FP16\n";
        return Precision::Half;
    }
    if (total_error_float <= tol / safety_factor) {
        if (verbose) std::cout << "  -> Selected: FP32\n";
        return Precision::Float;
    }

    if (verbose) std::cout << "  -> Selected: FP64\n";
    return Precision::Double;
}

#endif // PRECISION_H