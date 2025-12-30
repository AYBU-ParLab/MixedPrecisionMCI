#ifndef PRECISION_H
#define PRECISION_H

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <limits>
#include <queue>
#include <iostream>
#include <map>
#include <omp.h>
#include "parser.h"

const int FIXED_SEED = 42;

enum class Precision { Half, Float, Double, LongDouble };

struct Region {
    std::vector<double> bounds_min;
    std::vector<double> bounds_max;
    double error_estimate;
    int refinement_level;

    Region() : error_estimate(0), refinement_level(0) {}
    Region(const std::vector<double>& a, const std::vector<double>& b)
        : bounds_min(a), bounds_max(b), error_estimate(0), refinement_level(0) {}
    Region(double x1, double x2, double y1, double y2)
        : bounds_min{x1,y1}, bounds_max{x2,y2}, error_estimate(0), refinement_level(0) {}

    double volume() const {
        double v = 1.0;
        for (size_t i = 0; i < bounds_min.size(); ++i)
            v *= (bounds_max[i] - bounds_min[i]);
        return v;
    }
};


template<typename T>
inline T evaluate_postfix_fast(const std::vector<Token>& postfix, const T* vars, int dims,
                               const std::vector<std::string>* var_names = nullptr) {
    T stack[128];
    int sp = 0;

    // Build variable mapping if provided
    std::map<std::string, int> var_map;
    if (var_names != nullptr) {
        for (size_t i = 0; i < var_names->size() && i < (size_t)dims; ++i) {
            var_map[(*var_names)[i]] = static_cast<int>(i);
        }
    }

    for (const auto& t : postfix) {
        if (t.type == TokenType::Number) {
            stack[sp++] = static_cast<T>(std::stold(t.value));
        } else if (t.type == TokenType::Variable) {
            int i = -1;
            // First try variable map
            if (var_map.count(t.value)) {
                i = var_map[t.value];
            }
            // Fallback to alphabetic mapping
            else if (t.value.length() == 1) {
                char c = t.value[0];
                if (c >= 'a' && c <= 'z') i = c - 'a';
                else if (c >= 'A' && c <= 'Z') i = c - 'A';
            }
            stack[sp++] = (i>=0 && i<dims) ? vars[i] : T(0);
        } else if (t.type == TokenType::Operator) {
            T b = stack[--sp], a = stack[--sp];
            if (t.value=="+") stack[sp++] = a+b;
            else if (t.value=="-") stack[sp++] = a-b;
            else if (t.value=="*") stack[sp++] = a*b;
            else if (t.value=="/") stack[sp++] = b!=0?a/b:0;
            else if (t.value=="^") {
                if (std::floor(b)==b && std::abs(b)<=10) {
                    T r=1;
                    for(int i=0;i<(int)std::abs(b);++i) r*=a;
                    stack[sp++] = b>=0?r:T(1)/r;
                } else stack[sp++] = std::pow(a,b);
            }
        } else {
            T a = stack[--sp];
            if (t.value=="sin") stack[sp++] = std::sin(a);
            else if (t.value=="cos") stack[sp++] = std::cos(a);
            else if (t.value=="tan") stack[sp++] = std::tan(a);
            else if (t.value=="exp") stack[sp++] = std::exp(a);
            else if (t.value=="sqrt") stack[sp++] = a>=0?std::sqrt(a):0;
            else if (t.value=="ln") stack[sp++] = a>0?std::log(a):0;
            else if (t.value=="log") stack[sp++] = a>0?std::log10(a):0;
            else if (t.value=="abs") stack[sp++] = std::abs(a);
        }
    }
    return sp?stack[0]:T(0);
}

struct StatisticalMetrics {
    long double avg,var,grad,max_val,min_val,range,coefficient_of_variation;
    int samples_used;
};

inline StatisticalMetrics analyze_expression_fast(
    const std::vector<Token>& postfix,
    const std::vector<double>& minb,
    const std::vector<double>& maxb,
    int samples,
    const std::vector<std::string>* var_names = nullptr)
{
    StatisticalMetrics m{0,0,0,-1e300L,1e300L,0,0,0};
    int d = minb.size();

    long double sum=0,sumsq=0,maxg=0;
    int n=0;

#pragma omp parallel
    {
        std::mt19937 gen(FIXED_SEED + omp_get_thread_num()*1315423911);
        long double lsum=0,lsumsq=0,lmaxg=0,lmax=-1e300L,lmin=1e300L;
        int ln=0;

#pragma omp for
        for(int i=0;i<samples;i++){
            long double vars[32]={0};
            for(int k=0;k<d;k++){
                std::uniform_real_distribution<long double> dist(minb[k],maxb[k]);
                vars[k]=dist(gen);
            }

            long double v=evaluate_postfix_fast<long double>(postfix,vars,d,var_names);
            lsum+=v;
            lsumsq+=v*v;
            lmax=std::max(lmax,v);
            lmin=std::min(lmin,v);
            ln++;

            // Compute gradient for ALL dimensions (not just first 2)
            for(int k=0;k<d;k++){
                long double h=(maxb[k]-minb[k])*1e-2;
                vars[k]+=h;
                if(vars[k]<=maxb[k]){
                    long double vh=evaluate_postfix_fast<long double>(postfix,vars,d,var_names);
                    lmaxg=std::max(lmaxg,std::abs((vh-v)/h));
                }
                vars[k]-=h;
            }
        }

#pragma omp critical
        {
            sum+=lsum;
            sumsq+=lsumsq;
            maxg=std::max(maxg,lmaxg);
            m.max_val=std::max(m.max_val,lmax);
            m.min_val=std::min(m.min_val,lmin);
            n+=ln;
        }
    }

    if(n<2) return m;

    m.avg=sum/n;
    m.var=sumsq/n-m.avg*m.avg;
    m.grad=maxg;
    m.range=m.max_val-m.min_val;
    m.coefficient_of_variation=std::sqrt(std::abs(m.var))/(std::abs(m.avg)+1e-12L);
    m.samples_used=n;
    return m;
}

inline int count_operations(const std::vector<Token>& postfix) {
    return static_cast<int>(postfix.size());
}

inline bool has_risky_operations(const std::vector<Token>& postfix) {
    for (const auto& t : postfix) {
        if (t.type == TokenType::Operator && (t.value == "/" || t.value == "^")) return true;
        if (t.type == TokenType::Function && (t.value == "ln" || t.value == "log" || t.value == "tan")) return true;
    }
    return false;
}

inline Precision select_precision_for_term(
    const std::vector<Token>& postfix,
    const std::vector<double>& minb,
    const std::vector<double>& maxb,
    double tol,
    const std::string&,
    const std::vector<std::string>* var_names = nullptr)
{
    int dims = static_cast<int>(minb.size());
    int sample_count = (dims <= 4) ? 96 : (dims <= 7) ? 48 : 24;

    auto m = analyze_expression_fast(postfix, minb, maxb, sample_count, var_names);

    if (m.var == 0 && m.grad == 0) return Precision::Half;

    int ops = count_operations(postfix);
    bool has_risky = has_risky_operations(postfix);

    long double scale = std::max(std::abs(m.avg), std::sqrt(std::abs(m.var)));
    scale = std::max(scale, m.range * 0.5L);
    scale = std::max(scale, 1e-12L);

    long double cond = (m.grad * 1.0L) / (scale + 1e-12L);
    cond = std::max(1.0L, cond);

    if (has_risky) {
        if (m.range > 100.0L || cond > 100.0L) return Precision::Double;
        if (cond > 10.0L || ops > 20) return Precision::Float;
    }

    long double var_scale = std::sqrt(std::abs(m.var)) / (std::abs(m.avg) + 1e-12L);

    if (var_scale > 1.0L || cond > 50.0L) return Precision::Double;
    if (var_scale > 0.1L || cond > 5.0L || ops > 15) return Precision::Float;

    return Precision::Half;
}

inline Precision select_precision_for_region(
    const std::vector<Token>& postfix,
    const Region& region,
    double tolerance,
    const std::string& name,
    const std::vector<std::string>* var_names)
{
    auto m = analyze_expression_fast(postfix, region.bounds_min, region.bounds_max, 48, var_names);
    
    int ops = count_operations(postfix);
    bool has_risky = has_risky_operations(postfix);
    
    // CRITICAL: Use coefficient of variation - it's scale-invariant!
    long double cv = m.coefficient_of_variation;  // This is already computed!
    
    // High CV = unstable = needs FP64
    if (has_risky && cv > 0.1L) return Precision::Double;
    
    if (cv > 0.5L || m.grad > 10.0L) return Precision::Double;
    if (cv > 0.1L || m.grad > 2.0L || ops > 20) return Precision::Float;
    
    return Precision::Half;
}

inline std::vector<Region> adaptive_partition_nd(
    const std::vector<Token>& postfix,
    const Region& initial_region,
    double variance_threshold,
    double gradient_threshold,
    size_t max_regions,
    const std::vector<std::string>* var_names = nullptr)
{
    struct Item {
        Region region;
        long double score;
    };

    auto cmp = [](const Item& a, const Item& b) {
        return a.score < b.score;
    };

    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> pq(cmp);

    int dims = static_cast<int>(initial_region.bounds_min.size());
    int anal_samples = (dims <= 4) ? 64 : (dims <= 7) ? 32 : 16;

    auto m0 = analyze_expression_fast(
        postfix,
        initial_region.bounds_min,
        initial_region.bounds_max,
        anal_samples,
        var_names
    );

    long double s0 = (m0.var + 1e-12L) * (m0.grad + 1.0L);

    pq.push({initial_region, s0});

    std::vector<Region> result;

    // For very simple functions (low variance AND low gradient variation), use minimal regions
    // Check coefficient of variation: for linear functions like "a", CV is very low
    // CV = stddev / mean. For uniform "a" on [0,1]: mean=0.5, stddev=0.289, CV=0.577
    // But we're measuring sampling variance, not population variance
    long double cv = m0.coefficient_of_variation;

    // Linear functions have uniform behavior - single region is sufficient
    // Use very low gradient (derivative exists but function is smooth everywhere)
    if (cv < 1.5L && m0.grad < 5.0L) {
        return {initial_region};  // Single region is sufficient for smooth linear functions
    }

    size_t min_regions = (dims <= 3) ? 4 : (dims <= 5) ? 8 : 16;
    min_regions = std::min(min_regions, max_regions / 4);

    // For very simple functions (low initial score), stop early
    long double score_threshold = s0 * 0.001L;  // 0.1% of initial score
    if (s0 < 1e-6L) {
        // Extremely simple function, use single region
        return {initial_region};
    }

    while (!pq.empty() && result.size() + pq.size() < max_regions) {
        Item top = pq.top();
        pq.pop();

        // Stop if score is very low or significantly smaller than initial score
        if (top.score < 1e-9L ||
            (result.size() >= min_regions && top.score < score_threshold)) {
            result.push_back(top.region);
            continue;
        }

        const Region& r = top.region;

        int split_dim = 0;
        double max_extent = r.bounds_max[0] - r.bounds_min[0];
        for (int d = 1; d < dims; ++d) {
            double e = r.bounds_max[d] - r.bounds_min[d];
            if (e > max_extent) {
                max_extent = e;
                split_dim = d;
            }
        }

        if (max_extent < 1e-12) {
            result.push_back(r);
            continue;
        }

        double mid = 0.5 * (r.bounds_min[split_dim] + r.bounds_max[split_dim]);

        Region r1 = r;
        Region r2 = r;
        r1.bounds_max[split_dim] = mid;
        r2.bounds_min[split_dim] = mid;

        int split_samples = anal_samples / 2;
        split_samples = std::max(split_samples, 8);

        auto m1 = analyze_expression_fast(
            postfix, r1.bounds_min, r1.bounds_max, split_samples, var_names);
        auto m2 = analyze_expression_fast(
            postfix, r2.bounds_min, r2.bounds_max, split_samples, var_names);

        long double s1 = (m1.var + 1e-12L) * (m1.grad + 1.0L);
        long double s2 = (m2.var + 1e-12L) * (m2.grad + 1.0L);

        pq.push({r1, s1});
        pq.push({r2, s2});
    }

    while (!pq.empty() && result.size() < max_regions) {
        result.push_back(pq.top().region);
        pq.pop();
    }

    return result;
}
#endif
