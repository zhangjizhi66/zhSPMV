
# compiler
CXX = g++
NVCC = nvcc

# c compiler flags
CFLAGS += -O3
#CFLAGS += -g -Wall -Wextra -pedantic
#CFLAGS += -O0 -fno-inline
CFLAGS += -std=c++17
CFLAGS += -march=native
CFLAGS += -fopenmp
CFLAGS += -mavx -mavx2 -mfma
CFLAGS += -mavx512f -mavx512vl -mavx512dq -mavx512cd -mavx512bw

# nvcc compiler flags
NVCCFLAGS += -O3
NVCCFLAGS += -std=c++17
#NVCCFLAGS += -lcuda -lcudart

TARGET = spmv_avx256d spmv_avx512d spmv_xsimd256d spmv_xsimd512d spmv_xsimd256d_thread spmv_xsimd512d_thread spmv_xsimd256d_complex_test spmv_xsimd512d_complex_test spmv_cuda

all: $(TARGET)

spmv_avx256d: spmv_avx.cpp
	$(CXX) $(CFLAGS) -DVEC_LENGTH=256 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_avx512d: spmv_avx.cpp
	$(CXX) $(CFLAGS) -DVEC_LENGTH=512 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_xsimd256d: spmv_xsimd.cpp
	$(CXX) $(CFLAGS) -Iinclude -DVEC_LENGTH=256 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_xsimd512d: spmv_xsimd.cpp
	$(CXX) $(CFLAGS) -Iinclude -DVEC_LENGTH=512 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_xsimd256d_thread: spmv_xsimd_thread.cpp
	$(CXX) $(CFLAGS) -Iinclude -DVEC_LENGTH=256 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_xsimd512d_thread: spmv_xsimd_thread.cpp
	$(CXX) $(CFLAGS) -Iinclude -DVEC_LENGTH=512 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_xsimd256d_complex_test: spmv_xsimd_complex_test.cpp
	$(CXX) $(CFLAGS) -Iinclude -DVEC_LENGTH=256 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_xsimd512d_complex_test: spmv_xsimd_complex_test.cpp
	$(CXX) $(CFLAGS) -Iinclude -DVEC_LENGTH=512 -DDATATYPE=double -o $@ $<
	@echo "Build complete!"

spmv_cuda: spmv_cuda.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -o $@ $<
	@echo "Build complete!"

clean:
	rm -f $(TARGET)