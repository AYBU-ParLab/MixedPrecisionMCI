
CXX      := g++
CXXFLAGS := -O2 -std=c++17 -fopenmp

NVCC     := nvcc
NVCCFLAGS := -O2 -Xcompiler -fopenmp -std=c++17


CPP_SRCS := parser.cpp
CU_SRCS  := mci.cu

OBJ_CPP  := $(CPP_SRCS:.cpp=.o)
OBJ_CU   := $(CU_SRCS:.cu=.o)

TARGET   := mci

all: $(TARGET)

$(TARGET): $(OBJ_CPP) $(OBJ_CU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


run: $(TARGET)
	./$(TARGET)


clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(TARGET)
