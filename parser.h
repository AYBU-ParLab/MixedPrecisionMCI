#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <vector>
#include <stack>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

enum class TokenType { Number, Operator, Function, Variable, LeftParen, RightParen };

struct Token {
    TokenType type;
    std::string value;
    Token(TokenType t, const std::string& v); // declaration only
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

std::vector<Token> tokenize(const std::string& expr);
int precedence(const std::string& op);
bool is_right_associative(const std::string& op);
std::vector<Token> to_postfix(const std::vector<Token>& tokens);
std::vector<std::string> split_expression(const std::string& expr);


#endif // PARSER_H
