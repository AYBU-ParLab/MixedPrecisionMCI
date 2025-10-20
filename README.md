# GPU Accelerated Mixed Precision Monte Carlo Integration

CUDA-accelerated 2D Monte Carlo integration with automatic mixed-precision optimization.

## Features

- **GPU Acceleration** with CUDA batch processing
- **Mixed Precision**: Automatic float/double/long double selection
- **Multiple Strategies**: Term-wise and region-wise decomposition
- **CPU Comparison**: OpenMP parallel implementation
- **Expression Parser**: Supports `+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `log`, `ln`, `exp`, `sqrt`

## Requirements

- CUDA Toolkit (10.0+)
- NVIDIA GPU (compute capability 3.5+)
- G++ with C++17 support
- OpenMP

## Installation

```bash
make
```

## Usage

```bash
./mci
```

**Input Example:**
```
Expression: x*y + sin(x) + cos(y)
x-range [a b]: 0 3.14159
y-range [c d]: 0 3.14159
Total samples: 10000000
```

## How It Works

The framework automatically selects optimal precision for each term/region based on:
- Function variance
- Gradient magnitude
- Numerical stability
- Error tolerance (default: 1e-5)

**Precision Selection:**
- **Float**: Fast, suitable for smooth functions (~4x faster than double)
- **Double**: Default for moderate complexity
- **Long Double**: High precision (CPU only)

## Output

The program runs multiple integration strategies:
1. GPU Float Precision
2. GPU Double Precision
3. GPU Mixed Precision (term-wise)
4. GPU Mixed Precision (region-wise)
5. CPU comparisons

Each outputs timing breakdown and integration results.

## Monte Carlo Method

For function f(x,y) over [a,b] × [c,d]:

```
Integral ≈ (b-a)(d-c) × (1/N) × Σf(xi, yi)
```

## Customization

**Change tolerance:**
```cpp
input.tolerance = 1e-5;  // in main()
```

**Adjust OpenMP threads:**
```cpp
omp_set_num_threads(6);  // default is 6
```

## Contributors
- Ferhat Onur Özgan
- Berke Kabasakal