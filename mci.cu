#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <cmath>
#include <cctype>
#include <random>
#include <chrono>
#include <limits>
#include <omp.h>
#include <sstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "parser.h"

const int FIXED_SEED = 42; // Fixed seed for reproducibility

enum class Precision
{
    Float,
    Double,
    LongDouble
};
struct Region
{
    long double x1, x2, y1, y2;
};



template<typename T>
T evaluate_postfix(const std::vector<Token>& postfix, T x_val, T y_val) {
    std::stack<T> stack;
    for (const auto& token : postfix) {
        if (token.type == TokenType::Number) {
            stack.push(static_cast<T>(std::stold(token.value)));
        } else if (token.type == TokenType::Variable) {
            if (token.value == "x") stack.push(x_val);
            else if (token.value == "y") stack.push(y_val);
            else throw std::runtime_error("Unknown variable: " + token.value);
        } else if (token.type == TokenType::Operator) {
            T b = stack.top(); stack.pop();
            T a = stack.top(); stack.pop();
            if (token.value == "+") stack.push(a + b);
            else if (token.value == "-") stack.push(a - b);
            else if (token.value == "*") stack.push(a * b);
            else if (token.value == "/") { if (b == 0) throw std::runtime_error("Div by 0"); stack.push(a / b); }
            else if (token.value == "^") stack.push(std::pow(a, b));
        } else if (token.type == TokenType::Function) {
            T a = stack.top(); stack.pop();
            if (token.value == "sin") stack.push(std::sin(a));
            else if (token.value == "cos") stack.push(std::cos(a));
            else if (token.value == "log") { if (a <= 0) throw std::runtime_error("Log domain"); stack.push(std::log10(a)); }
            else if (token.value == "ln") { if (a <= 0) throw std::runtime_error("Ln domain"); stack.push(std::log(a)); }
            else if (token.value == "exp") stack.push(std::exp(a));
            else if (token.value == "sqrt") { if (a < 0) throw std::runtime_error("Sqrt domain"); stack.push(std::sqrt(a)); }
        }
    }
    return stack.top();
}

// ======================
// Precision Selection
// ======================
Precision select_precision(long double avg, long double grad, long double var,
                           double tol, const std::string& term,bool termwise = false)
{
    if (termwise && grad > 0 && var > 0 ) {
        // To detect linear function terms, uses normalized variance and gradient ratio 
        long double normalized_var = var / (avg * avg + 1e-10);
        if (normalized_var < 0.15 && (grad / std::abs(avg)) < 2.0) {
            std::cout << "\nPrecision Selection Analysis for term \"" << term << "\":\n";
            std::cout << "  Avg: " << avg << ", Var: " << var << ", Grad: " << grad << "\n";
            std::cout << "  Normalized Var: " << normalized_var << ", Grad/Avg: " << (grad/std::abs(avg)) << "\n";
            std::cout << "Selected: Float (simple term optimization) for \"" << term << "\"\n";
            return Precision::Float;
        }
    }
    if (var == 0 && grad == 0) {
        std::cout << "\nPrecision Selection Analysis for term \"" << term << "\":\n" << "  Avg: " << avg << ", Var: " << var << ", Grad: " << grad << "\n";
        return Precision::Float;
    }
    
    const long double eps_float = std::numeric_limits<float>::epsilon();
    const long double eps_double = std::numeric_limits<double>::epsilon();
    const long double eps_long_double = std::numeric_limits<long double>::epsilon();

    // Robust function value estimate
    long double max_val = std::max(std::abs(avg), std::sqrt(var));
    max_val = std::max(max_val, static_cast<long double>(1e-10));

    // Normalize gradient
    long double normalized_grad = std::max(grad, static_cast<long double>(1.0));

    // Condition number estimate
    long double condition_number =
        normalized_grad * max_val / std::max(std::abs(avg), static_cast<long double>(tol));
    condition_number = std::max(condition_number, static_cast<long double>(1.0));

    // Conservative op count
    long double operation_count = 10.0L;

    // Total rounding error = eps * value * cond * sqrt(ops)
    long double error_factor = max_val * condition_number * std::sqrt(operation_count);

    long double error_float = eps_float * error_factor;
    long double error_double = eps_double * error_factor;
    long double error_long_double = eps_long_double * error_factor;

    // Monte Carlo sampling contribution
    long double mc_error = std::sqrt(var);

    long double total_error_float = error_float + mc_error * eps_float;
    long double total_error_double = error_double + mc_error * eps_double;
    long double total_error_long_double = error_long_double + mc_error * eps_long_double;

    std::cout << "\nPrecision Selection Analysis for term \"" << term << "\":\n";
    std::cout << "  Avg: " << avg << ", Var: " << var << ", Grad: " << grad << "\n";
    std::cout << "  Condition number: " << condition_number << "\n";
    std::cout << "  Float error:       " << total_error_float << "\n";
    std::cout << "  Double error:      " << total_error_double << "\n";
    std::cout << "  Long Double error: " << total_error_long_double << "\n";
    std::cout << "  Tolerance:         " << tol << "\n";

    double safety_factor = termwise?10.0:0.1;

    if (total_error_float <= tol / safety_factor) {
        std::cout << "Selected: Float (sufficient precision) for \"" << term << "\"\n";
        return Precision::Float;
    }
    if (total_error_double <= tol / safety_factor) {
        std::cout << "Selected: Double (float insufficient) for \"" << term << "\"\n";
        return Precision::Double;
    }

    std::cout << "Selected: Long Double (higher precision required) for \"" << term << "\"\n";
    return Precision::LongDouble;
}


// ======================
// Average Value Estimate
// ======================
long double average_value(const std::vector<Token> &postfix,
                          long double a, long double b,
                          long double c, long double d)
{
    const int N = 10;
    long double sum = 0;
    int valid = 0;

#pragma omp parallel for reduction(+ : sum, valid) collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            long double x = a + (b - a) * i / (N - 1.0);
            long double y = c + (d - c) * j / (N - 1.0);

            // FIX: avoid singularity exactly at y=0
            if (y <= 1e-12L) y = 1e-12L;

            try {
                sum += std::abs(evaluate_postfix<long double>(postfix, x, y));
                valid++;
            } catch (...) {
            }
        }
    }
    return valid ? sum / valid : 0;
}


// ======================
// Variance Estimate
// ======================
long double estimate_variance(const std::vector<Token> &postfix,
                              long double a, long double b,
                              long double c, long double d, size_t samples)
{
    samples = std::min(samples, static_cast<size_t>(100));
    long double sum = 0, sum_sq = 0;
    int valid = 0;

#pragma omp parallel default(none) shared(postfix,a,b,c,d,samples) \
    reduction(+ : sum, sum_sq, valid)
    {
        std::mt19937 gen(FIXED_SEED);
        std::uniform_real_distribution<long double> dist_x(a, b);
        std::uniform_real_distribution<long double> dist_y(c, d);

#pragma omp for
        for (size_t i = 0; i < samples; ++i) {
            long double x = dist_x(gen);
            long double y = dist_y(gen);

            // FIX: avoid log singularity
            if (y <= 1e-12L) y = 1e-12L;

            try {
                long double val = evaluate_postfix<long double>(postfix, x, y);
                sum += val;
                sum_sq += val * val;
                valid++;
            } catch (...) {
            }
        }
    }

    if (valid < 2) return 0;
    long double mean = sum / valid;
    return (sum_sq / valid) - (mean * mean);
}


// ======================
// Gradient Estimate 
// ======================
long double estimate_gradient(const std::vector<Token> &postfix,
                              long double a, long double b,
                              long double c, long double d)
{
    const int N = 20; // FIX: higher resolution grid
    long double max_grad = 0;

#pragma omp parallel for reduction(max : max_grad) collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            long double x = a + (b - a) * i / (N - 1.0);
            long double y = c + (d - c) * j / (N - 1.0);

            // FIX: shift away from exact singularity
            if (y <= 1e-12L) y = 1e-12L;

            try {
                long double fx = evaluate_postfix<long double>(postfix, x, y);

                // X derivative
                if (i < N - 1) {
                    long double x_next = a + (b - a) * (i + 1) / (N - 1.0);
                    long double fx_next = evaluate_postfix<long double>(postfix, x_next, y);
                    long double grad_x = std::abs((fx_next - fx) / (x_next - x));
                    max_grad = std::max(max_grad, grad_x);
                }

                // Y derivative
                if (j < N - 1) {
                    long double y_next = c + (d - c) * (j + 1) / (N - 1.0);
                    if (y_next <= 1e-12L) y_next = 1e-12L;
                    long double fy_next = evaluate_postfix<long double>(postfix, x, y_next);
                    long double grad_y = std::abs((fy_next - fx) / (y_next - y));
                    max_grad = std::max(max_grad, grad_y);
                }
            } catch (...) {
            }
        }
    }
    return max_grad;
}


// Custom device string comparison function
__device__ bool device_strcmp(const char *str1, const char *str2)
{
    while (*str1 != '\0' && *str2 != '\0')
    {
        if (*str1 != *str2)
            return false;
        ++str1;
        ++str2;
    }
    return *str1 == '\0' && *str2 == '\0';
}

// CUDA device function for evaluating postfix expression
template <typename T>
__device__ T evaluate_postfix_device(const TokenType *types, const char *values, int token_count, T x_val, T y_val)
{
    T stack[100]; // Fixed-size stack for GPU
    int stack_pos = 0;

    for (int i = 0; i < token_count; ++i)
    {
        TokenType type = types[i];

        if (type == TokenType::Number)
        {
            // Parse number from string
            T num = 0;
            bool decimal = false;
            T decimal_pos = 1;

            for (int j = 0; values[i * 20 + j] != '\0' && j < 20; ++j)
            {
                char c = values[i * 20 + j];
                if (c == '.')
                {
                    decimal = true;
                }
                else if (c >= '0' && c <= '9')
                {
                    if (decimal)
                    {
                        decimal_pos *= 0.1;
                        num += (c - '0') * decimal_pos;
                    }
                    else
                    {
                        num = num * 10 + (c - '0');
                    }
                }
            }
            stack[stack_pos++] = num;
        }
        else if (type == TokenType::Variable)
        {
            char var = values[i * 20];
            if (var == 'x')
                stack[stack_pos++] = x_val;
            else if (var == 'y')
                stack[stack_pos++] = y_val;
            // Ignore other variables
        }
        else if (type == TokenType::Operator)
        {
            char op = values[i * 20];
            T b = stack[--stack_pos];
            T a = stack[--stack_pos];

            if (op == '+')
                stack[stack_pos++] = a + b;
            else if (op == '-')
                stack[stack_pos++] = a - b;
            else if (op == '*')
                stack[stack_pos++] = a * b;
            else if (op == '/')
            {
                if (b == 0)
                    return 0; // Error handling: return 0 for division by zero
                stack[stack_pos++] = a / b;
            }
            else if (op == '^')
                stack[stack_pos++] = pow(a, b);
        }
        else if (type == TokenType::Function)
        {
            T a = stack[--stack_pos];

            // Check first three chars to determine function
            const char *func = &values[i * 20];

            if (device_strcmp(func, "sin"))
                stack[stack_pos++] = sin(a);
            else if (device_strcmp(func, "cos"))
                stack[stack_pos++] = cos(a);
            else if (device_strcmp(func, "log"))
            {
                if (a <= 0)
                    return 0; // Error handling
                stack[stack_pos++] = log10(a);
            }
            else if (device_strcmp(func, "ln"))
            {
                if (a <= 0)
                    return 0; // Error handling
                stack[stack_pos++] = log(a);
            }
            else if (device_strcmp(func, "exp"))
                stack[stack_pos++] = exp(a);
            else if (device_strcmp(func, "sqrt"))
            {
                if (a < 0)
                    return 0; // Error handling
                stack[stack_pos++] = sqrt(a);
            }
        }
    }

    return stack_pos > 0 ? stack[stack_pos - 1] : 0;
}

// Helper function to prepare CUDA data from tokens
void prepare_cuda_data(const std::vector<Token> &postfix, TokenType **d_types, char **d_values)
{
    int token_count = postfix.size();

    // Allocate host memory
    TokenType *h_types = new TokenType[token_count];
    char *h_values = new char[token_count * 20]; // Assume max 20 chars per token value

    // Fill host data
    for (int i = 0; i < token_count; ++i)
    {
        h_types[i] = postfix[i].type;

        // Copy value with zero padding
        strncpy(&h_values[i * 20], postfix[i].value.c_str(), 19); // Removed std::
        h_values[i * 20 + std::min(static_cast<int>(postfix[i].value.length()), 19)] = '\0';
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(d_types, token_count * sizeof(TokenType)));
    CUDA_CHECK(cudaMalloc(d_values, token_count * 20 * sizeof(char)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(*d_types, h_types, token_count * sizeof(TokenType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_values, h_values, token_count * 20 * sizeof(char), cudaMemcpyHostToDevice));

    // Free host memory
    delete[] h_types;
    delete[] h_values;
}

// CUDA kernel for Monte Carlo integration
template <typename T>
__device__ T generate_random(curandState &state, T a, T b);

template <>
__device__ float generate_random(curandState &state, float a, float b)
{
    return a + (b - a) * curand_uniform(&state);
}

template <>
__device__ double generate_random(curandState &state, double a, double b)
{
    return a + (b - a) * curand_uniform_double(&state);
}

// CUDA kernel for batch region processing
template <typename T>
__global__ void monte_carlo_regions_kernel(
    TokenType *types, char *values, int token_count,
    T *region_bounds, // [x1,x2,y1,y2] for each region
    int num_regions,
    unsigned long long samples_per_thread,
    T *results,
    unsigned long long *valid_counts,
    unsigned long long seed)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int region_idx = blockIdx.y;

    if (region_idx >= num_regions)
        return;

    // Get region bounds
    T xa = region_bounds[region_idx * 4 + 0];
    T xb = region_bounds[region_idx * 4 + 1];
    T yc = region_bounds[region_idx * 4 + 2];
    T yd = region_bounds[region_idx * 4 + 3];

    // Initialize random number generator
    curandState state;
    curand_init(seed + thread_idx + region_idx * 10000, 0, 0, &state);

    T sum = 0;
    unsigned long long valid = 0;

    for (unsigned long long i = 0; i < samples_per_thread; ++i)
    {
        T x = generate_random(state, xa, xb);
        T y = generate_random(state, yc, yd);

        T value = evaluate_postfix_device<T>(types, values, token_count, x, y);

        if (isfinite(value))
        {
            sum += value;
            valid++;
        }
    }

    // Store results using grid-stride indexing
    int grid_stride = gridDim.x * blockDim.x;
    results[region_idx * grid_stride + thread_idx] = sum;
    valid_counts[region_idx * grid_stride + thread_idx] = valid;
}

// Batch region processing function
template <typename T>
std::vector<T> monte_carlo_integrate_regions_cuda_batch(
    const std::vector<Region> &regions,
    size_t samples_per_region,
    const std::vector<Token> &postfix)
{
    int num_regions = regions.size();
    if (num_regions == 0)
        return {};

    // CUDA configuration
    const int threadsPerBlock = 256;
    const int blocksPerRegion = 32; // Adjust based on your GPU

    dim3 grid(blocksPerRegion, num_regions);
    dim3 block(threadsPerBlock);

    unsigned long long samples_per_thread =
        (samples_per_region + blocksPerRegion * threadsPerBlock - 1) / (blocksPerRegion * threadsPerBlock);

    // Prepare token data (single copy for all regions)
    TokenType *d_types = nullptr;
    char *d_values = nullptr;
    prepare_cuda_data(postfix, &d_types, &d_values);

    // Prepare region bounds
    std::vector<T> h_bounds(num_regions * 4);
    for (int i = 0; i < num_regions; i++)
    {
        h_bounds[i * 4 + 0] = static_cast<T>(regions[i].x1);
        h_bounds[i * 4 + 1] = static_cast<T>(regions[i].x2);
        h_bounds[i * 4 + 2] = static_cast<T>(regions[i].y1);
        h_bounds[i * 4 + 3] = static_cast<T>(regions[i].y2);
    }

    T *d_bounds;
    cudaMalloc(&d_bounds, num_regions * 4 * sizeof(T));
    cudaMemcpy(d_bounds, h_bounds.data(), num_regions * 4 * sizeof(T), cudaMemcpyHostToDevice);

    // Allocate result arrays
    const size_t results_size = grid.x * grid.y * block.x;
    T *d_results;
    unsigned long long *d_valid_counts;
    cudaMalloc(&d_results, results_size * sizeof(T));
    cudaMalloc(&d_valid_counts, results_size * sizeof(unsigned long long));

    // Generate seed
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Launch single kernel for all regions
    monte_carlo_regions_kernel<T><<<grid, block>>>(
        d_types, d_values, postfix.size(),
        d_bounds, num_regions,
        samples_per_thread,
        d_results, d_valid_counts,
        seed);

    // Copy results back
    std::vector<T> h_results(results_size);
    std::vector<unsigned long long> h_valid_counts(results_size);

    cudaMemcpy(h_results.data(), d_results, results_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid_counts.data(), d_valid_counts, results_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Reduce results per region
    std::vector<T> final_results(num_regions);
    const int points_per_region = grid.x * block.x;

    for (int region = 0; region < num_regions; region++)
    {
        T sum = 0;
        unsigned long long valid_total = 0;

        for (int i = 0; i < points_per_region; i++)
        {
            const int idx = region * points_per_region + i;
            sum += h_results[idx];
            valid_total += h_valid_counts[idx];
        }

        if (valid_total > 0)
        {
            T area = (static_cast<T>(regions[region].x2) - static_cast<T>(regions[region].x1)) *
                     (static_cast<T>(regions[region].y2) - static_cast<T>(regions[region].y1));
            final_results[region] = area * sum / valid_total;
        }
        else
        {
            final_results[region] = 0;
        }
    }

    // Cleanup
    cudaFree(d_types);
    cudaFree(d_values);
    cudaFree(d_bounds);
    cudaFree(d_results);
    cudaFree(d_valid_counts);

    return final_results;
}

template <typename T>
T monte_carlo_integrate_2d(size_t samples, T a, T b, T c, T d, const std::vector<Token> &postfix)
{
    T sum = 0;
    size_t valid_samples = 0;
#pragma omp parallel default(none) shared(samples, a, b, c, d, postfix, sum, valid_samples)
    {
        std::mt19937 gen(FIXED_SEED);
        std::uniform_real_distribution<T> dist_x(a, b);
        std::uniform_real_distribution<T> dist_y(c, d);
        T local_sum = 0;
        size_t local_valid = 0;

#pragma omp for
        for (size_t i = 0; i < samples; ++i)
        {
            T x = dist_x(gen);
            T y = dist_y(gen);
            try
            {
                local_sum += evaluate_postfix<T>(postfix, x, y);
                local_valid++;
            }
            catch (...)
            {
            }
        }

#pragma omp atomic
        sum += local_sum;
#pragma omp atomic
        valid_samples += local_valid;
    }
    if (valid_samples == 0)
        return 0;
    T area = (b - a) * (d - c);
    return area * sum / valid_samples; // Use valid_samples instead of samples
}
template <typename T>
T monte_carlo_integrate_2d_cpu(size_t samples, T a, T b, T c, T d, const std::vector<Token> &postfix)
{
    std::mt19937 gen(FIXED_SEED);
    std::uniform_real_distribution<T> dist_x(a, b);
    std::uniform_real_distribution<T> dist_y(c, d);

    T sum = 0;
    size_t valid = 0;

    for (size_t i = 0; i < samples; ++i)
    {
        T x = dist_x(gen);
        T y = dist_y(gen);

        try
        {
            T value = evaluate_postfix<T>(postfix, x, y);
            if (std::isfinite(value))
            {
                sum += value;
                valid++;
            }
        }
        catch (...)
        {
        }
    }

    if (valid == 0)
        return 0;
    T area = (b - a) * (d - c);
    return area * sum / valid;
}

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::cerr << "CUDA error (" << __FILE__ << ":" << __LINE__   \
                      << ") - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

// Add this after the includes, before other code
struct InputData
{
    std::string expr;
    long double xa, xb, yc, yd;
    size_t total_samples;
    double tolerance;
};

template <typename T>
__global__ void monte_carlo_batch_kernel(
    TokenType **d_types_array,
    char **d_values_array,
    int *d_token_counts,
    int num_terms,
    T *d_results,
    unsigned long long *d_valid_counts,
    T xa, T xb, T yc, T yd,
    unsigned long long samples_per_thread,
    unsigned long long seed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int term_idx = blockIdx.y;

    if (term_idx >= num_terms)
        return;

    curandState state;
    curand_init(seed + tid, term_idx, 0, &state);

    T sum = 0;
    unsigned long long valid = 0;

    for (unsigned long long i = 0; i < samples_per_thread; ++i)
    {
        T x = generate_random(state, xa, xb);
        T y = generate_random(state, yc, yd);

        T value = evaluate_postfix_device<T>(
            d_types_array[term_idx],
            d_values_array[term_idx],
            d_token_counts[term_idx],
            x, y);

        if (isfinite(value))
        {
            sum += value;
            valid++;
        }
    }

    // Store results using grid-stride indexing
    const int grid_stride = gridDim.x * blockDim.x;
    d_results[term_idx * grid_stride + tid] = sum;
    d_valid_counts[term_idx * grid_stride + tid] = valid;
}

template <typename T>
std::vector<T> monte_carlo_integrate_2d_cuda_batch(
    size_t samples,
    T xa, T xb, T yc, T yd,
    const std::vector<std::vector<Token>> &all_postfixes)
{
    int num_terms = all_postfixes.size();
    if (num_terms == 0)
        return {};

    // CUDA configuration
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int threadsPerBlock = 256;
    const int blocksPerTerm = prop.multiProcessorCount * 4; // Use multiple blocks per SM

    dim3 grid(blocksPerTerm, num_terms);
    dim3 block(threadsPerBlock);

    unsigned long long samples_per_thread =
        (samples + blocksPerTerm * threadsPerBlock - 1) / (blocksPerTerm * threadsPerBlock);

    // Device arrays for token data
    std::vector<TokenType *> d_types_vec(num_terms);
    std::vector<char *> d_values_vec(num_terms);
    std::vector<int> token_counts(num_terms);

    // Prepare device memory for each term
    for (int i = 0; i < num_terms; i++)
    {
        token_counts[i] = all_postfixes[i].size();
        prepare_cuda_data(all_postfixes[i], &d_types_vec[i], &d_values_vec[i]);
    }

    // Allocate and copy array of pointers
    TokenType **d_types_array;
    char **d_values_array;
    int *d_token_counts;

    cudaMalloc(&d_types_array, num_terms * sizeof(TokenType *));
    cudaMalloc(&d_values_array, num_terms * sizeof(char *));
    cudaMalloc(&d_token_counts, num_terms * sizeof(int));

    cudaMemcpy(d_types_array, d_types_vec.data(), num_terms * sizeof(TokenType *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_array, d_values_vec.data(), num_terms * sizeof(char *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_token_counts, token_counts.data(), num_terms * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate result arrays
    const size_t results_size = grid.x * grid.y * block.x;
    T *d_results;
    unsigned long long *d_valid_counts;
    cudaMalloc(&d_results, results_size * sizeof(T));
    cudaMalloc(&d_valid_counts, results_size * sizeof(unsigned long long));

    // Generate seed
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Launch kernel
    monte_carlo_batch_kernel<T><<<grid, block>>>(
        d_types_array,
        d_values_array,
        d_token_counts,
        num_terms,
        d_results,
        d_valid_counts,
        xa, xb, yc, yd,
        samples_per_thread,
        seed);

    // Allocate and copy results back to host
    std::vector<T> h_results(results_size);
    std::vector<unsigned long long> h_valid_counts(results_size);

    cudaMemcpy(h_results.data(), d_results, results_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid_counts.data(), d_valid_counts, results_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Reduce results on CPU
    std::vector<T> final_results(num_terms);
    const T area = (xb - xa) * (yd - yc);
    const int points_per_term = grid.x * block.x;

    for (int term = 0; term < num_terms; term++)
    {
        T sum = 0;
        unsigned long long valid_total = 0;

        for (int i = 0; i < points_per_term; i++)
        {
            const int idx = term * points_per_term + i;
            sum += h_results[idx];
            valid_total += h_valid_counts[idx];
        }

        final_results[term] = valid_total > 0 ? (area * sum / valid_total) : 0;
    }

    // Cleanup
    for (auto ptr : d_types_vec)
        cudaFree(ptr);
    for (auto ptr : d_values_vec)
        cudaFree(ptr);
    cudaFree(d_types_array);
    cudaFree(d_values_array);
    cudaFree(d_token_counts);
    cudaFree(d_results);
    cudaFree(d_valid_counts);

    return final_results;
}
int main()
{
    std::cout << "Mixed Precision Monte Carlo Integration (CUDA + CPU)\n";

    // 1) Initialize CUDA once
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found. Exiting...\n";
        return EXIT_FAILURE;
    }
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Using CUDA device: " << deviceProp.name
              << " with " << deviceProp.multiProcessorCount << " SMs\n";
    CHECK_CUDA(cudaSetDevice(0));

    // 2) Read input once
    InputData input;
    input.tolerance = 1e-5;

    std::cout << "Expression (e.g., x*y + sin(x) + cos(y)): ";
    std::getline(std::cin, input.expr);
    std::cout << "x-range [a b]: ";
    std::cin >> input.xa >> input.xb;
    std::cout << "y-range [c d]: ";
    std::cin >> input.yc >> input.yd;
    std::cout << "Total samples: ";
    std::cin >> input.total_samples;
    if (input.xa > input.xb)
        std::swap(input.xa, input.xb);
    if (input.yc > input.yd)
        std::swap(input.yc, input.yd);

    // 3) Pre-parse terms into postfix form once
    auto terms = split_expression(input.expr);
    std::vector<std::vector<Token>> postfixes;
    postfixes.reserve(terms.size());
    for (const auto &term : terms)
    {
        postfixes.push_back(to_postfix(tokenize(term)));
    }
    std::vector<std::vector<Token>> warmup_postfix = {postfixes[0]};
    monte_carlo_integrate_2d_cuda_batch<float>(
        1000,
        static_cast<float>(input.xa), static_cast<float>(input.xb),
        static_cast<float>(input.yc), static_cast<float>(input.yd),
        warmup_postfix);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Float Precision Strategy ---
    std::cout << "\n=== GPU Float Precision Strategy ===\n";
    {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        cudaEvent_t startF, stopF, memStart, memStop;
        CHECK_CUDA(cudaEventCreate(&startF));
        CHECK_CUDA(cudaEventCreate(&stopF));
        CHECK_CUDA(cudaEventCreate(&memStart));
        CHECK_CUDA(cudaEventCreate(&memStop));
        
        CHECK_CUDA(cudaEventRecord(memStart, 0));
        CHECK_CUDA(cudaEventRecord(startF, 0));

        // Use batch processing for all terms in float precision
        std::vector<float> results = monte_carlo_integrate_2d_cuda_batch<float>(
            input.total_samples,
            static_cast<float>(input.xa), static_cast<float>(input.xb),
            static_cast<float>(input.yc), static_cast<float>(input.yd),
            postfixes);

        CHECK_CUDA(cudaEventRecord(stopF, 0));
        CHECK_CUDA(cudaEventRecord(memStop, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        float ms_kernel = 0.0f, ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, startF, stopF));
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, memStart, memStop));
        
        long double total_gpu_f = 0.0L;
        for (size_t i = 0; i < results.size(); ++i)
        {
            printf("[float]   Term \"%-20s\" => %.8f\n",
                   terms[i].c_str(), results[i]);
            total_gpu_f += results[i];
        }
        
        auto end_total = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_total - start_total).count();
        double t_kernel = ms_kernel / 1000.0;
        double t_mem = (ms_total - ms_kernel) / 1000.0;
        
        printf("\nTiming Breakdown:\n");
        printf("  Kernel Execution: %.6f s\n", t_kernel);
        printf("  Memory Transfer:  %.6f s\n", t_mem);
        printf("  Total GPU Time:   %.6f s\n", ms_total / 1000.0);
        printf("  Wall Clock Time:  %.6f s\n", wall_time);
        printf("GPU (float batch) = %.8Lf\n", total_gpu_f);
        
        CHECK_CUDA(cudaEventDestroy(startF));
        CHECK_CUDA(cudaEventDestroy(stopF));
        CHECK_CUDA(cudaEventDestroy(memStart));
        CHECK_CUDA(cudaEventDestroy(memStop));
    }
    monte_carlo_integrate_2d_cuda_batch<double>(
        1000,
        static_cast<double>(input.xa), static_cast<double>(input.xb),
        static_cast<double>(input.yc), static_cast<double>(input.yd),
        warmup_postfix);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Double Precision Strategy ---
    std::cout << "\n=== GPU Double Precision Strategy ===\n";
    {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        cudaEvent_t startD, stopD, memStart, memStop;
        CHECK_CUDA(cudaEventCreate(&startD));
        CHECK_CUDA(cudaEventCreate(&stopD));
        CHECK_CUDA(cudaEventCreate(&memStart));
        CHECK_CUDA(cudaEventCreate(&memStop));
        
        CHECK_CUDA(cudaEventRecord(memStart, 0));
        CHECK_CUDA(cudaEventRecord(startD, 0));

        // Use batch processing for all terms in double precision
        std::vector<double> results = monte_carlo_integrate_2d_cuda_batch<double>(
            input.total_samples,
            static_cast<double>(input.xa), static_cast<double>(input.xb),
            static_cast<double>(input.yc), static_cast<double>(input.yd),
            postfixes);

        CHECK_CUDA(cudaEventRecord(stopD, 0));
        CHECK_CUDA(cudaEventRecord(memStop, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        float ms_kernel = 0.0f, ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, startD, stopD));
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, memStart, memStop));
        
        long double total_gpu_d = 0.0L;
        for (size_t i = 0; i < results.size(); ++i)
        {
            printf("[double]  Term \"%-20s\" => %.8f\n",
                   terms[i].c_str(), results[i]);
            total_gpu_d += results[i];
        }
        
        auto end_total = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_total - start_total).count();
        double t_kernel = ms_kernel / 1000.0;
        double t_mem = (ms_total - ms_kernel) / 1000.0;
        
        printf("\nTiming Breakdown:\n");
        printf("  Kernel Execution: %.6f s\n", t_kernel);
        printf("  Memory Transfer:  %.6f s\n", t_mem);
        printf("  Total GPU Time:   %.6f s\n", ms_total / 1000.0);
        printf("  Wall Clock Time:  %.6f s\n", wall_time);
        printf("GPU (double batch) = %.8Lf\n", total_gpu_d);
        
        CHECK_CUDA(cudaEventDestroy(startD));
        CHECK_CUDA(cudaEventDestroy(stopD));
        CHECK_CUDA(cudaEventDestroy(memStart));
        CHECK_CUDA(cudaEventDestroy(memStop));
    }
    monte_carlo_integrate_2d_cuda_batch<float>(
        1000,
        static_cast<float>(input.xa), static_cast<float>(input.xb),
        static_cast<float>(input.yc), static_cast<float>(input.yd),
        warmup_postfix);
    monte_carlo_integrate_2d_cuda_batch<double>(
        1000,
        static_cast<double>(input.xa), static_cast<double>(input.xb),
        static_cast<double>(input.yc), static_cast<double>(input.yd),
        warmup_postfix);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // --- Mixed Precision Strategy ---
    std::cout << "\n=== GPU Mixed Precision Strategy (term wise) ===\n";
    {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        cudaEvent_t startM, stopM, memStart, memStop;
        CHECK_CUDA(cudaEventCreate(&startM));
        CHECK_CUDA(cudaEventCreate(&stopM));
        CHECK_CUDA(cudaEventCreate(&memStart));
        CHECK_CUDA(cudaEventCreate(&memStop));

        // Precision selection timing
        auto prec_start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<Token>> float_terms, double_terms;
        std::vector<size_t> float_indices, double_indices;

        // Terms classification
        for (size_t i = 0; i < terms.size(); ++i)
        {
            auto tokens = tokenize(terms[i]);
            auto postfix = to_postfix(tokens);
            long double avg = average_value(postfix, input.xa, input.xb, input.yc, input.yd);
            long double var = estimate_variance(postfix, input.xa, input.xb, input.yc, input.yd, input.total_samples);
            long double grad = estimate_gradient(postfix, input.xa, input.xb, input.yc, input.yd);
            Precision prec = select_precision(avg, grad, var, input.tolerance, terms[i],true);

            if (prec == Precision::Float)
            {
                float_terms.push_back(postfix);
                float_indices.push_back(i);
            }
            else
            {
                double_terms.push_back(postfix);
                double_indices.push_back(i);
            }
        }
        
        auto prec_end = std::chrono::high_resolution_clock::now();
        double prec_time = std::chrono::duration<double>(prec_end - prec_start).count();

        CHECK_CUDA(cudaEventRecord(memStart, 0));
        CHECK_CUDA(cudaEventRecord(startM, 0));

        long double total_gpu_m = 0.0L;
        std::vector<double> mixed_results;

        // Process float terms
        if (!float_terms.empty())
        {
            std::vector<float> results = monte_carlo_integrate_2d_cuda_batch<float>(
                input.total_samples,
                static_cast<float>(input.xa), static_cast<float>(input.xb),
                static_cast<float>(input.yc), static_cast<float>(input.yd),
                float_terms);

            for (size_t i = 0; i < results.size(); ++i)
            {
                printf("[float]   Term \"%-20s\" => %.8f\n",
                       terms[float_indices[i]].c_str(), results[i]);
                total_gpu_m += results[i];
            }
        }

        // Process double terms
        if (!double_terms.empty())
        {
            std::vector<double> results = monte_carlo_integrate_2d_cuda_batch<double>(
                input.total_samples,
                static_cast<double>(input.xa), static_cast<double>(input.xb),
                static_cast<double>(input.yc), static_cast<double>(input.yd),
                double_terms);

            for (size_t i = 0; i < results.size(); ++i)
            {
                printf("[double]  Term \"%-20s\" => %.14f\n",
                       terms[double_indices[i]].c_str(), results[i]);
                total_gpu_m += results[i];
            }
        }
        
        CHECK_CUDA(cudaEventRecord(stopM, 0));
        CHECK_CUDA(cudaEventRecord(memStop, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        float ms_kernel = 0.0f, ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, startM, stopM));
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, memStart, memStop));
        
        auto end_total = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_total - start_total).count();
        double t_kernel = ms_kernel / 1000.0;
        double t_mem = (ms_total - ms_kernel) / 1000.0;
        
        printf("\nTiming Breakdown:\n");
        printf("  Precision Selection: %.6f s\n", prec_time);
        printf("  Kernel Execution:    %.6f s\n", t_kernel);
        printf("  Memory Transfer:     %.6f s\n", t_mem);
        printf("  Total GPU Time:      %.6f s\n", ms_total / 1000.0);
        printf("  Wall Clock Time:     %.6f s\n", wall_time);
        printf("Result gpu mixed (Term wise): %.14Lf\n", total_gpu_m);
        
        CHECK_CUDA(cudaEventDestroy(startM));
        CHECK_CUDA(cudaEventDestroy(stopM));
        CHECK_CUDA(cudaEventDestroy(memStart));
        CHECK_CUDA(cudaEventDestroy(memStop));
    }
    
    std::vector<Region> f_regions, d_regions;
    monte_carlo_integrate_regions_cuda_batch<float>(
                f_regions, 1000, postfixes[0]);
    monte_carlo_integrate_regions_cuda_batch<double>(
                d_regions, 1000, postfixes[0]);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // === REGION-WISE ANALYSIS ===
    long double xmid = (input.xa + input.xb) / 2.0L, ymid = (input.yc + input.yd) / 2.0L;
    std::vector<Region> regions = {{input.xa, xmid, input.yc, ymid}, {xmid, input.xb, input.yc, ymid}, {input.xa, xmid, ymid, input.yd}, {xmid, input.xb, ymid, input.yd}};
    size_t samples_per_region = input.total_samples / regions.size();

    std::cout << "\n=== GPU Region-wise Mixed Precision ===\n";
    {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        cudaEvent_t start, stop, memStart, memStop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventCreate(&memStart));
        CHECK_CUDA(cudaEventCreate(&memStop));

        // Precision selection timing
        auto prec_start = std::chrono::high_resolution_clock::now();
        
        long double gpu_regions = 0.0L;
        size_t reg_float = 0, reg_double = 0;

        // Group regions by precision
        std::vector<Region> float_regions, double_regions;
        std::vector<size_t> float_indices, double_indices;

        std::cout << "GPU Region Analysis:\n";
        for (size_t i = 0; i < regions.size(); ++i)
        {
            const auto &R = regions[i];
            auto postfix = to_postfix(tokenize(input.expr));
            long double avg = average_value(postfix, R.x1, R.x2, R.y1, R.y2);
            long double var = estimate_variance(postfix, R.x1, R.x2, R.y1, R.y2, samples_per_region);
            long double grad = estimate_gradient(postfix, R.x1, R.x2, R.y1, R.y2);
            Precision prec = select_precision(avg, grad, var, input.tolerance, input.expr + " (Region " + std::to_string(i+1) + ")");

            printf("  Region[%zu]: Avg=%8.4Lf, Var=%8.4Le, Grad=%8.4Le -> ", i, avg, var, grad);

            if (prec == Precision::Float)
            {
                printf("FLOAT\n");
                float_regions.push_back(R);
                float_indices.push_back(i);
                reg_float++;
            }
            else
            {
                printf("DOUBLE\n");
                double_regions.push_back(R);
                double_indices.push_back(i);
                reg_double++;
            }
        }
        
        auto prec_end = std::chrono::high_resolution_clock::now();
        double prec_time = std::chrono::duration<double>(prec_end - prec_start).count();

        CHECK_CUDA(cudaEventRecord(memStart, 0));
        CHECK_CUDA(cudaEventRecord(start, 0));

        // Process float regions in batch
        if (!float_regions.empty())
        {
            auto postfix = to_postfix(tokenize(input.expr));
            auto f_results = monte_carlo_integrate_regions_cuda_batch<float>(
                float_regions, samples_per_region, postfix);
            for (size_t i = 0; i < f_results.size(); ++i)
            {
                printf("  [float] Region[%zu] => %.8f\n", float_indices[i], f_results[i]);
                gpu_regions += f_results[i];
            }
        }

        // Process double regions in batch
        if (!double_regions.empty())
        {
            auto postfix = to_postfix(tokenize(input.expr));
            auto d_results = monte_carlo_integrate_regions_cuda_batch<double>(
                double_regions, samples_per_region, postfix);
            for (size_t i = 0; i < d_results.size(); ++i)
            {
                printf("  [double] Region[%zu] => %.14f\n", double_indices[i], d_results[i]);
                gpu_regions += d_results[i];
            }
        }
        
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventRecord(memStop, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        float ms_kernel = 0.0f, ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, start, stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, memStart, memStop));
        
        auto end_total = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_total - start_total).count();
        double t_kernel = ms_kernel / 1000.0;
        double t_mem = (ms_total - ms_kernel) / 1000.0;
        double eff = (reg_float + 2.0 * reg_double) / (2.0 * regions.size());

        printf("\nTiming Breakdown:\n");
        printf("  Precision Selection: %.6f s\n", prec_time);
        printf("  Kernel Execution:    %.6f s\n", t_kernel);
        printf("  Memory Transfer:     %.6f s\n", t_mem);
        printf("  Total GPU Time:      %.6f s\n", ms_total / 1000.0);
        printf("  Wall Clock Time:     %.6f s\n", wall_time);
        printf("Result: %.8Lf | Efficiency: %.2f | F/D: %zu/%zu\n",
               gpu_regions, eff, reg_float, reg_double);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaEventDestroy(memStart));
        CHECK_CUDA(cudaEventDestroy(memStop));
    }
    // --- CPU Full Float Precision ---
    std::cout << "\n=== CPU Full Float Precision ===\n";
    {
        long double total_cpu_f = 0.0L;
        double start_cpu_f = omp_get_wtime();
        for (const auto &post : postfixes)
        {
            total_cpu_f += monte_carlo_integrate_2d<float>(
                input.total_samples,
                static_cast<float>(input.xa), static_cast<float>(input.xb),
                static_cast<float>(input.yc), static_cast<float>(input.yd),
                post);
        }
        double end_cpu_f = omp_get_wtime();
        double t_cpu_f = end_cpu_f - start_cpu_f;
        std::cout << "CPU (float term-wise) = " << total_cpu_f
                  << " in " << t_cpu_f << " s\n";
    }

    // --- CPU Full Double Precision ---
    std::cout << "\n=== CPU Full Double Precision ===\n";
    {
        long double total_cpu_d = 0.0L;
        double start_cpu_d = omp_get_wtime();
        for (const auto &post : postfixes)
        {
            total_cpu_d += monte_carlo_integrate_2d<double>(
                input.total_samples,
                static_cast<double>(input.xa), static_cast<double>(input.xb),
                static_cast<double>(input.yc), static_cast<double>(input.yd),
                post);
        }
        double end_cpu_d = omp_get_wtime();
        double t_cpu_d = end_cpu_d - start_cpu_d;
        std::cout << "CPU (double term-wise) = " << total_cpu_d
                  << " in " << t_cpu_d << " s\n";
    }

    omp_set_num_threads(6);
    std::cout << "\n--- CPU Term-wise [" << omp_get_max_threads() << "]---\n";
    long double total_parallel = 0;
    double start_parallel = omp_get_wtime();

    // Term analizi paralel olarak yap
    struct TermInfo {
        Precision prec;
        double result;
        size_t idx;
    };
    std::vector<TermInfo> term_info(terms.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < terms.size(); ++i) {
        auto postfix = to_postfix(tokenize(terms[i]));
        long double avg = average_value(postfix, input.xa, input.xb, input.yc, input.yd);
        long double var = estimate_variance(postfix, input.xa, input.xb, input.yc, input.yd, input.total_samples);
        long double grad = estimate_gradient(postfix, input.xa, input.xb, input.yc, input.yd);
        term_info[i].prec = select_precision(avg, grad, var, input.tolerance, terms[i],true);
        term_info[i].result = 0.0;
        term_info[i].idx = i;
    }

    // Monte Carlo integrasyonu paralel yap
    #pragma omp parallel for reduction(+:total_parallel) schedule(guided)
    for (size_t i = 0; i < terms.size(); ++i) {
        auto& info = term_info[i];
        auto postfix = to_postfix(tokenize(terms[info.idx]));
        double sub_start = omp_get_wtime();

        if (info.prec == Precision::Float) {
            info.result = monte_carlo_integrate_2d<float>(
                input.total_samples,
                static_cast<float>(input.xa), static_cast<float>(input.xb),
                static_cast<float>(input.yc), static_cast<float>(input.yd),
                postfix);
            total_parallel += info.result;
            #pragma omp critical
            std::cout << "Term \"" << terms[info.idx] << "\" | float | Result: " << info.result 
                     << " | Time: " << (omp_get_wtime() - sub_start) << " sec\n";
        }
        else if (info.prec == Precision::Double) {
            info.result = monte_carlo_integrate_2d<double>(
                input.total_samples,
                input.xa, input.xb, input.yc, input.yd,
                postfix);
            total_parallel += info.result;
            #pragma omp critical
            std::cout << "Term \"" << terms[info.idx] << "\" | double | Result: " << info.result 
                     << " | Time: " << (omp_get_wtime() - sub_start) << " sec\n";
        }
        else {
            info.result = monte_carlo_integrate_2d<long double>(
                input.total_samples,
                input.xa, input.xb, input.yc, input.yd,
                postfix);
            total_parallel += info.result;
            #pragma omp critical
            std::cout << "Term \"" << terms[info.idx] << "\" | long double | Result: " << info.result 
                     << " | Time: " << (omp_get_wtime() - sub_start) << " sec\n";
        }
    }

    double end_parallel = omp_get_wtime();
    double parallel_time = end_parallel - start_parallel;
    std::cout << "Result: " << total_parallel << " | Total Time: " << parallel_time << " sec\n";

    // CPU region-wise (OpenMP)
    std::cout << "\n== CPU Region-wise Integration (OpenMP) ==\n";
    long double cpu_regions = 0.0L;
    double start_cpu_regions = omp_get_wtime();
#pragma omp parallel for reduction(+ : cpu_regions)
    for (size_t i = 0; i < regions.size(); ++i)
    {
        const auto &R = regions[i];
        auto tokens = tokenize(input.expr);
        auto postfix = to_postfix(tokens);
        long double avg = average_value(postfix, R.x1, R.x2, R.y1, R.y2);
        long double var = estimate_variance(postfix, R.x1, R.x2, R.y1, R.y2, samples_per_region);
        long double grad = estimate_gradient(postfix, R.x1, R.x2, R.y1, R.y2);
        Precision prec = select_precision(avg, grad, var, input.tolerance, input.expr + " (Region " + std::to_string(i+1) + ")");

        long double res = 0.0L;
        if (prec == Precision::Float)
        {
            std::cout << "[float]   ";
            res = monte_carlo_integrate_2d<float>(samples_per_region, R.x1, R.x2, R.y1, R.y2, postfix);
        }
        else if (prec == Precision::Double)
        {
            std::cout << "[double]  ";
            res = monte_carlo_integrate_2d<double>(samples_per_region, R.x1, R.x2, R.y1, R.y2, postfix);
        }
        else
        {
            std::cout << "[long double] ";
            res = monte_carlo_integrate_2d<long double>(samples_per_region, R.x1, R.x2, R.y1, R.y2, postfix);
        }
        std::cout << "Region " << (i + 1) << " => " << res << "\n";
        cpu_regions += res;
    }
    double end_cpu_regions = omp_get_wtime();
    double cpu_reg_s = end_cpu_regions - start_cpu_regions;

    std::cout << "Result: " << cpu_regions << " in " << cpu_reg_s << " s\n";

    CHECK_CUDA(cudaDeviceReset());
    return EXIT_SUCCESS;
}