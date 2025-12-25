# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture - adjust for your GPU
# Common options:
#   sm_86: RTX 3070/3080/3090, A6000
#   sm_89: RTX 4070/4080/4090, L40
#   sm_90: H100
CUDA_ARCH = sm_86

# Compiler flags - OPTIMIZED
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 --use_fast_math \
             -Xcompiler -fopenmp -Xcompiler -march=native \
             --ptxas-options=-v \
             -lineinfo \
             -DNDEBUG

CXX_FLAGS = -O3 -std=c++17 -fopenmp -march=native

# Include directories
INCLUDES = -I.

# Source files
CU_SOURCES = cuda_integration.cu
CPP_SOURCES = main.cpp parser.cpp
HEADERS = parser.h precision.h cuda_integration.h

# Object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Executable name
TARGET = mci_optimized

# Default target
all: $(TARGET)

# Link everything
$(TARGET): $(CU_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lcudart -lcurand

# Compile CUDA files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ files with NVCC to ensure CUDA headers are visible
%.o: %.cpp $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(CU_OBJECTS) $(CPP_OBJECTS) $(TARGET) *.ptx

# Run the program with sample input
run: $(TARGET)
	@echo "x*y + sin(x)*cos(y)" | ./$(TARGET)

# Check CUDA device info
device-info:
	nvidia-smi
	@echo ""
	@echo "Detailed device information:"
	@$(NVCC) -arch=$(CUDA_ARCH) --run cuda_device_query.cu 2>/dev/null || echo "Build device query tool with: nvcc -o device_query cuda_device_query.cu && ./device_query"

# Profile with Nsight Compute (requires appropriate permissions)
profile: $(TARGET)
	ncu --set full --export profile_report ./$(TARGET)

# Profile with Nsight Systems
profile-sys: $(TARGET)
	nsys profile --stats=true -o profile_timeline ./$(TARGET)

# Build with debug symbols
debug: NVCC_FLAGS = -arch=$(CUDA_ARCH) -G -g -O0 -Xcompiler -fopenmp
debug: clean $(TARGET)

# Benchmark mode - disable all printing
benchmark: NVCC_FLAGS += -DBENCHMARK_MODE
benchmark: clean $(TARGET)

# Check for common issues
check:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null && echo "✓ nvcc found" || echo "✗ nvcc not found"
	@nvidia-smi > /dev/null 2>&1 && echo "✓ NVIDIA driver found" || echo "✗ NVIDIA driver not found"
	@echo ""
	@echo "Checking compute capability..."
	@$(NVCC) --help | grep -A 10 "gpu-architecture" || true

# Generate PTX for inspection
ptx: cuda_integration.cu
	$(NVCC) -arch=$(CUDA_ARCH) -ptx cuda_integration.cu -o cuda_integration.ptx

# Phony targets
.PHONY: all clean run device-info profile profile-sys debug benchmark check ptx

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the optimized executable (default)"
	@echo "  clean        - Remove object files and executable"
	@echo "  run          - Build and run with sample input"
	@echo "  device-info  - Show CUDA device information"
	@echo "  profile      - Profile with Nsight Compute"
	@echo "  profile-sys  - Profile with Nsight Systems"
	@echo "  debug        - Build with debug symbols"
	@echo "  benchmark    - Build in benchmark mode"
	@echo "  check        - Check CUDA installation"
	@echo "  ptx          - Generate PTX assembly for inspection"
	@echo ""
	@echo "Adjust CUDA_ARCH in Makefile for your GPU:"
	@echo "  RTX 3070/3080/3090: sm_86"
	@echo "  RTX 4070/4080/4090: sm_89"
	@echo "  H100: sm_90"