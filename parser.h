#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>
#include <cmath>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

enum class TokenType { Number, Operator, Function, Variable, LeftParen, RightParen };

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v) : type(t), value(v) {}
};

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) abort(); } while(0)

std::vector<Token> tokenize(const std::string& expr);
int precedence(const std::string& op);
bool is_right_associative(const std::string& op);
std::vector<Token> to_postfix(const std::vector<Token>& tokens);
std::vector<std::string> split_expression(const std::string& expr);

// Extract unique variable names from expression (supports up to 10 dimensions)
inline std::vector<std::string> extract_variables(const std::string& expr) {
    std::vector<std::string> vars;
    std::set<std::string> vars_set;  // For duplicate checking

    // Common math functions to exclude
    std::set<std::string> functions = {"sin", "cos", "tan", "exp", "sqrt", "ln", "log", "abs"};

    size_t i = 0;
    while (i < expr.length()) {
        char c = expr[i];

        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            std::string name;
            name += c;

            // Check if it's a multi-character identifier
            size_t j = i + 1;
            while (j < expr.length() &&
                   ((expr[j] >= 'a' && expr[j] <= 'z') ||
                    (expr[j] >= 'A' && expr[j] <= 'Z') ||
                    (expr[j] >= '0' && expr[j] <= '9'))) {
                name += expr[j];
                j++;
            }

            // Only add if it's not a function name and not already added
            if (functions.find(name) == functions.end()) {
                if (vars_set.insert(name).second) {
                    // First time seeing this variable - add in order of appearance
                    vars.push_back(name);
                }
            }
            i = j;
        } else {
            i++;
        }
    }

    // Return variables in order of first appearance (NO SORTING!)
    return vars;
}

inline int detect_dimensions(const std::string& expr) {
    auto vars = extract_variables(expr);
    return vars.empty() ? 1 : std::min(10, (int)vars.size());  // Max 10 dimensions
}

struct CompiledExpr {
    std::vector<TokenType> types;
    std::vector<float> constants;
    std::vector<int> var_indices;
    std::vector<int> op_codes;
    int expr_length;
    int dimensions;
};

CompiledExpr compile_expression(const std::vector<Token>& postfix, int dims,
                                const std::vector<std::string>* var_names = nullptr);

template <typename T>
inline T evaluate_compiled(const CompiledExpr& c, const T* vars, int max_dims = 32) {
    T stack[64];
    int sp = 0;

    for (int i = 0; i < c.expr_length; ++i) {
        switch (c.types[i]) {
            case TokenType::Number:
                stack[sp++] = (T)c.constants[i];
                break;
            case TokenType::Variable: {
                int idx = c.var_indices[i];
                stack[sp++] = (idx >= 0 && idx < max_dims) ? vars[idx] : 0;
                break;
            }
            case TokenType::Operator: {
                T b = stack[--sp];
                T a = stack[--sp];
                switch (c.op_codes[i]) {
                    case 0: stack[sp++] = a + b; break;
                    case 1: stack[sp++] = a - b; break;
                    case 2: stack[sp++] = a * b; break;
                    case 3: stack[sp++] = b ? a / b : 0; break;
                    case 4: stack[sp++] = pow(a, b); break;
                }
                break;
            }
            case TokenType::Function: {
                T a = stack[--sp];
                switch (c.op_codes[i]) {
                    case 10: stack[sp++] = sin(a); break;
                    case 11: stack[sp++] = cos(a); break;
                    case 12: stack[sp++] = a > 0 ? log10(a) : 0; break;
                    case 13: stack[sp++] = a > 0 ? log(a) : 0; break;
                    case 14: stack[sp++] = exp(a); break;
                    case 15: stack[sp++] = a >= 0 ? sqrt(a) : 0; break;
                    case 16: stack[sp++] = tan(a); break;
                    case 17: stack[sp++] = abs(a); break;
                }
                break;
            }
            default: break;
        }
    }
    return stack[0];
}

template <typename T>
inline T evaluate_compiled(const CompiledExpr& c, T x, T y) {
    T vars[32] = {x, y};
    return evaluate_compiled(c, vars, 2);
}

#endif
