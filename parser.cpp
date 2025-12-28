#include "parser.h"
#include <cctype>
#include <algorithm>
#include <stack>
#include <stdexcept>
#include <map>

std::vector<Token> tokenize(const std::string& expr) {
    std::vector<Token> tokens;
    size_t i = 0;
    
    while (i < expr.length()) {
        char c = expr[i];
        
        if (isspace(c)) { ++i; continue; }
        
        if (isdigit(c) || c == '.') {
            std::string num;
            while (i < expr.length() && (isdigit(expr[i]) || expr[i] == '.' || 
                   expr[i] == 'e' || expr[i] == 'E' || 
                   (i > 0 && (expr[i-1] == 'e' || expr[i-1] == 'E') && 
                   (expr[i] == '+' || expr[i] == '-')))) {
                num += expr[i++];
            }
            tokens.emplace_back(TokenType::Number, num);
        }
        else if (isalpha(c)) {
            std::string name;
            while (i < expr.length() && isalpha(expr[i])) name += expr[i++];
            
            if (name == "sin" || name == "cos" || name == "log" || name == "ln" || 
                name == "exp" || name == "sqrt" || name == "tan" || name == "abs")
                tokens.emplace_back(TokenType::Function, name);
            else
                tokens.emplace_back(TokenType::Variable, name);
        }
        else if (std::string("+-*/^").find(c) != std::string::npos) {
            if ((c == '-' || c == '+') && (i == 0 || 
                (i > 0 && (expr[i-1] == '(' || 
                 std::string("+-*/^").find(expr[i-1]) != std::string::npos)))) {
                if (c == '-') {
                    tokens.emplace_back(TokenType::Number, "0");
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
        if (stack.top().type == TokenType::LeftParen) 
            throw std::runtime_error("Mismatched parentheses");
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
        
        if ((c == '+' || c == '-') && paren == 0 && i > 0) {
            char prev = expr[i-1];
            bool is_binary = !(prev == '+' || prev == '-' || prev == '*' || 
                              prev == '/' || prev == '^' || prev == '(' || 
                              prev == 'e' || prev == 'E');
            
            if (is_binary) {
                if (!current.empty()) {
                    terms.push_back(current);
                    current.clear();
                }
                if (c == '-') current += c;
                continue;
            }
        }
        current += c;
    }
    
    if (!current.empty()) terms.push_back(current);
    return terms;
}

CompiledExpr compile_expression(const std::vector<Token>& postfix, int dims,
                                const std::vector<std::string>* var_names) {
    CompiledExpr compiled;
    compiled.expr_length = postfix.size();
    compiled.dimensions = dims;

    // Build variable name to index mapping
    std::map<std::string, int> var_map;
    if (var_names != nullptr) {
        for (size_t i = 0; i < var_names->size() && i < (size_t)dims; ++i) {
            var_map[(*var_names)[i]] = static_cast<int>(i);
        }
    }

    for (const auto& token : postfix) {
        compiled.types.push_back(token.type);

        if (token.type == TokenType::Number) {
            compiled.constants.push_back(std::stof(token.value));
            compiled.var_indices.push_back(-1);
            compiled.op_codes.push_back(-1);
        }
        else if (token.type == TokenType::Variable) {
            compiled.constants.push_back(0.0f);
            int idx = -1;

            // First try using the variable map
            if (var_map.count(token.value)) {
                idx = var_map[token.value];
            }
            // Fallback to old single-character mapping
            else if (token.value.length() == 1) {
                char c = token.value[0];
                if (c >= 'a' && c <= 'z') idx = c - 'a';
                else if (c >= 'A' && c <= 'Z') idx = c - 'A';
            }
            compiled.var_indices.push_back(idx);
            compiled.op_codes.push_back(-1);
        }
        else if (token.type == TokenType::Operator) {
            compiled.constants.push_back(0.0f);
            compiled.var_indices.push_back(-1);
            if (token.value == "+") compiled.op_codes.push_back(0);
            else if (token.value == "-") compiled.op_codes.push_back(1);
            else if (token.value == "*") compiled.op_codes.push_back(2);
            else if (token.value == "/") compiled.op_codes.push_back(3);
            else if (token.value == "^") compiled.op_codes.push_back(4);
            else compiled.op_codes.push_back(-1);
        }
        else if (token.type == TokenType::Function) {
            compiled.constants.push_back(0.0f);
            compiled.var_indices.push_back(-1);
            if (token.value == "sin") compiled.op_codes.push_back(10);
            else if (token.value == "cos") compiled.op_codes.push_back(11);
            else if (token.value == "log") compiled.op_codes.push_back(12);
            else if (token.value == "ln") compiled.op_codes.push_back(13);
            else if (token.value == "exp") compiled.op_codes.push_back(14);
            else if (token.value == "sqrt") compiled.op_codes.push_back(15);
            else if (token.value == "tan") compiled.op_codes.push_back(16);
            else if (token.value == "abs") compiled.op_codes.push_back(17);
            else compiled.op_codes.push_back(-1);
        }
        else {
            compiled.constants.push_back(0.0f);
            compiled.var_indices.push_back(-1);
            compiled.op_codes.push_back(-1);
        }
    }
    
    return compiled;
}