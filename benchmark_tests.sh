#!/bin/bash

# 20 Different Test Functions with Various Ranges and Sample Counts
# Sample counts range from 10^6 to 10^11

echo "======================================================================"
echo "  COMPREHENSIVE BENCHMARK: 20 TEST FUNCTIONS"
echo "  Testing Xorshift Random Number Generator Performance"
echo "======================================================================"

# Test 1: Simple polynomial (1D) - 1M samples
echo -e "\n[TEST 1] Simple polynomial x^3 - 2*x^2 + x"
./mci_optimized --func "x^3 - 2*x^2 + x" --bounds "-2:2" --sample 1000000

# Test 2: Trigonometric (2D) - 10M samples
echo -e "\n[TEST 2] sin(x)*cos(y)"
./mci_optimized --func "sin(x)*cos(y)" --bounds "0:3.14159,0:3.14159" --sample 10000000

# Test 3: Exponential decay (3D) - 100M samples
echo -e "\n[TEST 3] exp(-x-y-z)"
./mci_optimized --func "exp(-x-y-z)" --bounds "0:5,0:5,0:5" --sample 100000000

# Test 4: Gaussian-like (2D) - 500M samples
echo -e "\n[TEST 4] exp(-(x^2 + y^2))"
./mci_optimized --func "exp(-(x^2 + y^2))" --bounds "-3:3,-3:3" --sample 500000000

# Test 5: Oscillating function (1D) - 1B samples
echo -e "\n[TEST 5] sin(10*x) * exp(-x)"
./mci_optimized --func "sin(10*x) * exp(-x)" --bounds "0:10" --sample 1000000000

# Test 6: Complex 4D function - 2B samples
echo -e "\n[TEST 6] sin(x+y)*cos(z+w)"
./mci_optimized --func "sin(x+y)*cos(z+w)" --bounds "0:6.28,0:6.28,0:6.28,0:6.28" --sample 2000000000

# Test 7: Rational function (2D) - 5M samples
echo -e "\n[TEST 7] 1/(1 + x^2 + y^2)"
./mci_optimized --func "1/(1 + x^2 + y^2)" --bounds "-5:5,-5:5" --sample 5000000

# Test 8: Logarithmic (3D) - 50M samples
echo -e "\n[TEST 8] log(1 + x + y + z)"
./mci_optimized --func "log(1 + x + y + z)" --bounds "0:10,0:10,0:10" --sample 50000000

# Test 9: Power function (1D) - 10B samples
echo -e "\n[TEST 9] x^5 - 3*x^3 + 2*x"
./mci_optimized --func "x^5 - 3*x^3 + 2*x" --bounds "-2:2" --sample 10000000000

# Test 10: High oscillation (2D) - 200M samples
echo -e "\n[TEST 10] sin(20*x) * sin(20*y)"
./mci_optimized --func "sin(20*x) * sin(20*y)" --bounds "0:3.14159,0:3.14159" --sample 200000000

# Test 11: Mixed operations (3D) - 1B samples
echo -e "\n[TEST 11] exp(x) * sin(y) * log(1+z)"
./mci_optimized --func "exp(x) * sin(y) * log(1+z)" --bounds "0:2,0:6.28,0:10" --sample 1000000000

# Test 12: Polynomial chaos (4D) - 5B samples
echo -e "\n[TEST 12] x^2 + y^3 - z^4 + w^5"
./mci_optimized --func "x^2 + y^3 - z^4 + w^5" --bounds "-1:1,-1:1,-1:1,-1:1" --sample 5000000000

# Test 13: Circular symmetry (2D) - 300M samples
echo -e "\n[TEST 13] sqrt(x^2 + y^2)"
./mci_optimized --func "sqrt(x^2 + y^2)" --bounds "-10:10,-10:10" --sample 300000000

# Test 14: Damped oscillation (1D) - 20B samples
echo -e "\n[TEST 14] exp(-0.5*x) * cos(5*x)"
./mci_optimized --func "exp(-0.5*x) * cos(5*x)" --bounds "0:20" --sample 20000000000

# Test 15: Product of trig functions (3D) - 800M samples
echo -e "\n[TEST 15] sin(x)*cos(y)*tan(z)"
./mci_optimized --func "sin(x)*cos(y)*tan(z)" --bounds "0:3.14,0:3.14,-1:1" --sample 800000000

# Test 16: Exponential polynomial (2D) - 50B samples
echo -e "\n[TEST 16] exp(x*y) / (1 + x^2 + y^2)"
./mci_optimized --func "exp(x*y) / (1 + x^2 + y^2)" --bounds "-2:2,-2:2" --sample 50000000000

# Test 17: Hyperbolic functions (3D) - 400M samples
echo -e "\n[TEST 17] 1/(1 + x^2) + 1/(1 + y^2) + 1/(1 + z^2)"
./mci_optimized --func "1/(1 + x^2) + 1/(1 + y^2) + 1/(1 + z^2)" --bounds "-5:5,-5:5,-5:5" --sample 400000000

# Test 18: Radial function (4D) - 100B samples
echo -e "\n[TEST 18] exp(-(x^2 + y^2 + z^2 + w^2))"
./mci_optimized --func "exp(-(x^2 + y^2 + z^2 + w^2))" --bounds "-3:3,-3:3,-3:3,-3:3" --sample 100000000000

# Test 19: Complex polynomial (2D) - 2.5M samples
echo -e "\n[TEST 19] x^4 - 6*x^2*y^2 + y^4"
./mci_optimized --func "x^4 - 6*x^2*y^2 + y^4" --bounds "-2:2,-2:2" --sample 2500000

# Test 20: Ultra-heavy (1D) - 75B samples
echo -e "\n[TEST 20] sin(x) + cos(2*x) + sin(3*x) + cos(4*x)"
./mci_optimized --func "sin(x) + cos(2*x) + sin(3*x) + cos(4*x)" --bounds "0:6.28318" --sample 75000000000

echo -e "\n======================================================================"
echo "  ALL TESTS COMPLETED"
echo "======================================================================"
