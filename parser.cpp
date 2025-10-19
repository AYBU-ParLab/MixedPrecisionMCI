
#include "parser.h"
#include <cmath>
#include <stack>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

Token::Token(TokenType t, const std::string& v) : type(t), value(v) {}
// Helper function for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::vector<Token> tokenize(const std::string& expr) {
    std::vector<Token> tokens;
    size_t i = 0;
    while (i < expr.length()) {
        char c = expr[i];

        if (isspace(c)) { ++i; continue; }

        if (isdigit(c) || c == '.') {
            std::string num;
            while (i < expr.length() && (isdigit(expr[i]) || expr[i] == '.')) num += expr[i++];
            tokens.emplace_back(TokenType::Number, num);
        }
        else if (isalpha(c)) {
            std::string name;
            while (i < expr.length() && isalpha(expr[i])) name += expr[i++];
            if (name == "sin" || name == "cos" || name == "log" || name == "ln" || name == "exp" || name == "sqrt")
                tokens.emplace_back(TokenType::Function, name);
            else
                tokens.emplace_back(TokenType::Variable, name);
        }
        else if (std::string("+-*/^").find(c) != std::string::npos) {
            // Handle unary minus/plus
            if ((c == '-' || c == '+') && (i == 0 || 
                (i > 0 && (expr[i-1] == '(' || 
                 std::string("+-*/^").find(expr[i-1]) != std::string::npos)))) {
                
                if (c == '-') {
                    // Insert 0 before unary minus to handle it as binary operation (0-x)
                    tokens.emplace_back(TokenType::Number, "0");
                }
                // For unary plus, we can just skip it
                if (c == '-') {
                    tokens.emplace_back(TokenType::Operator, std::string(1, c));
                }
            } else {
                tokens.emplace_back(TokenType::Operator, std::string(1, c));
            }
            ++i;
        }
        else if (c == '(') {
            tokens.emplace_back(TokenType::LeftParen, "(");
            ++i;
        }
        else if (c == ')') {
            tokens.emplace_back(TokenType::RightParen, ")");
            ++i;
        }
        else {
            throw std::runtime_error(std::string("Invalid character: ") + c);
        }
    }

    return tokens;
}

int precedence(const std::string& op) {
    if (op == "+" || op == "-") return 1;
    if (op == "*" || op == "/") return 2;
    if (op == "^") return 3;
    return 0;
}

bool is_right_associative(const std::string& op) {
    return op == "^";
}

std::vector<Token> to_postfix(const std::vector<Token>& tokens) {
    std::vector<Token> output;
    std::stack<Token> stack;

    for (const auto& token : tokens) {
        if (token.type == TokenType::Number || token.type == TokenType::Variable) {
            output.push_back(token);
        }
        else if (token.type == TokenType::Function) {
            stack.push(token);
        }
        else if (token.type == TokenType::Operator) {
            while (!stack.empty() && stack.top().type == TokenType::Operator &&
                   ((precedence(stack.top().value) > precedence(token.value)) ||
                    (precedence(stack.top().value) == precedence(token.value) &&
                     !is_right_associative(token.value)))) {
                output.push_back(stack.top());
                stack.pop();
            }
            stack.push(token);
        }
        else if (token.type == TokenType::LeftParen) {
            stack.push(token);
        }
        else if (token.type == TokenType::RightParen) {
            while (!stack.empty() && stack.top().type != TokenType::LeftParen) {
                output.push_back(stack.top());
                stack.pop();
            }
            if (stack.empty()) throw std::runtime_error("Mismatched parentheses");
            stack.pop();
            if (!stack.empty() && stack.top().type == TokenType::Function) {
                output.push_back(stack.top());
                stack.pop();
            }
        }
    }

    while (!stack.empty()) {
        if (stack.top().type == TokenType::LeftParen) throw std::runtime_error("Mismatched parentheses");
        output.push_back(stack.top());
        stack.pop();
    }

    return output;
}



std::vector<std::string> split_expression(const std::string& expr) {
    std::vector<std::string> terms;
    std::string current;
    int paren = 0;
    
    for (size_t i = 0; i < expr.length(); ++i) {
        char c = expr[i];
        
        if (c == '(') paren++;
        else if (c == ')') paren--;
        
        // Check for +/- operators that are not inside parentheses
        if ((c == '+' || c == '-') && paren == 0) {
            // Skip if this is the first character or after another operator (unary operator)
            bool is_binary_operator = (i > 0);
            
            if (is_binary_operator) {
                char prev = expr[i-1];
                // Check if previous char is an operator or left parenthesis (making this a unary + or -)
                if (prev == '+' || prev == '-' || prev == '*' || prev == '/' || prev == '^' || prev == '(' || prev == 'e' || prev == 'E') {
                    is_binary_operator = false;  // This is likely a sign or part of scientific notation
                }
            }
            
            if (is_binary_operator) {
                if (!current.empty()) {
                    terms.push_back(current);
                    current.clear();
                }
                
                // If it's a minus sign, we add it to the next term
                if (c == '-') {
                    current += c;
                }
                // We skip the + sign as it's implied between terms
                continue;
            }
        }
        
        // Add the character to the current term
        current += c;
    }
    
    if (!current.empty()) {
        terms.push_back(current);
    }
    
    return terms;
}