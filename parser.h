#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>
#include <stack>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <set>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

enum class TokenType { Number, Operator, Function, Variable, LeftParen, RightParen };

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v) : type(t), value(v) {}
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Tokenization and parsing
std::vector<Token> tokenize(const std::string& expr);
int precedence(const std::string& op);
bool is_right_associative(const std::string& op);
std::vector<Token> to_postfix(const std::vector<Token>& tokens);
std::vector<std::string> split_expression(const std::string& expr);

// Auto-detect dimensionality from expression
inline int detect_dimensions(const std::string& expr) {
    std::set<char> vars;
    for (size_t i = 0; i < expr.length(); ++i) {
        if (isalpha(expr[i])) {
            // Check if it's a standalone variable, not part of a function name
            bool is_var = true;
            if (i > 0 && isalpha(expr[i-1])) is_var = false;
            if (i < expr.length()-1 && isalpha(expr[i+1])) is_var = false;
            
            if (is_var) {
                char c = expr[i];
                if (c == 'x' || c == 'y' || c == 'z' || c == 'w') {
                    vars.insert(c);
                }
            }
        }
    }
    return vars.empty() ? 1 : static_cast<int>(vars.size());
}

// Compiled expression structure for optimized evaluation
struct CompiledExpr {
    std::vector<TokenType> types;
    std::vector<float> constants;
    std::vector<int> var_indices;
    std::vector<int> op_codes;
    int expr_length;
    int dimensions;  // Track dimensionality
};

CompiledExpr compile_expression(const std::vector<Token>& postfix, int dims);

// CPU-side evaluation of compiled expression (multi-dimensional)
template <typename T>
inline T evaluate_compiled(const CompiledExpr& compiled, const T* vars) {
    T stack[64];
    int sp = 0;
    
    for (int i = 0; i < compiled.expr_length; ++i) {
        if (compiled.types[i] == TokenType::Number) {
            stack[sp++] = static_cast<T>(compiled.constants[i]);
        }
        else if (compiled.types[i] == TokenType::Variable) {
            int var_idx = compiled.var_indices[i];
            if (var_idx >= 0 && var_idx < compiled.dimensions) {
                stack[sp++] = vars[var_idx];
            } else {
                stack[sp++] = 0;
            }
        }
        else if (compiled.types[i] == TokenType::Operator) {
            int opcode = compiled.op_codes[i];
            if (opcode >= 0 && opcode <= 4) {
                T b = stack[--sp];
                T a = stack[--sp];
                switch (opcode) {
                    case 0: stack[sp++] = a + b; break;
                    case 1: stack[sp++] = a - b; break;
                    case 2: stack[sp++] = a * b; break;
                    case 3: stack[sp++] = (b != 0) ? (a / b) : 0; break;
                    case 4: stack[sp++] = std::pow(a, b); break;
                }
            }
        }
        else if (compiled.types[i] == TokenType::Function) {
            int opcode = compiled.op_codes[i];
            if (opcode >= 10 && opcode <= 17) {
                T a = stack[--sp];
                switch (opcode) {
                    case 10: stack[sp++] = std::sin(a); break;
                    case 11: stack[sp++] = std::cos(a); break;
                    case 12: stack[sp++] = (a > 0) ? std::log10(a) : 0; break;
                    case 13: stack[sp++] = (a > 0) ? std::log(a) : 0; break;
                    case 14: stack[sp++] = std::exp(a); break;
                    case 15: stack[sp++] = (a >= 0) ? std::sqrt(a) : 0; break;
                    case 16: stack[sp++] = std::tan(a); break;
                    case 17: stack[sp++] = std::abs(a); break;
                }
            }
        }
    }
    
    return (sp > 0) ? stack[0] : 0;
}

// Overload for backward compatibility (2D)
template <typename T>
inline T evaluate_compiled(const CompiledExpr& compiled, T x_val, T y_val) {
    T vars[4] = {x_val, y_val, 0, 0};
    return evaluate_compiled(compiled, vars);
}

#endif // PARSER_H