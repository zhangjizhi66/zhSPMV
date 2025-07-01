
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ctime>
#include <chrono>
#include "csr.h"

#define DATATYPE double

// Parameters
#define DEFAULT_NLOOPS 1000
#define DEFAULT_NTHREADS 1
#define DEFAULT_PRECISION 1e-14

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

template <typename T>
__global__ void SPMV_GPU(const T *nnz_val, const int *col_idx, const int *row_ptr, const T *x, T *y) {
    int irow = blockIdx.x;
    int global_idx = row_ptr[irow] + threadIdx.x;  // idx in the matrix
    int local_idx = threadIdx.x;  // idx in the row
    int row_length = row_ptr[irow + 1] - row_ptr[irow];

    extern __shared__ T prefix_sum[];
    if ( local_idx < row_length ) {
        prefix_sum[local_idx] = nnz_val[global_idx] * x[col_idx[global_idx]];
    }
    __syncthreads();

    for (int i = 1; i <= blockDim.x; i *= 2) {
        if ( local_idx - i >= 0 ) {
            prefix_sum[local_idx] += prefix_sum[local_idx - i];
        }
        __syncthreads();
    }

    if ( local_idx == row_length - 1 )
        y[irow] = prefix_sum[local_idx];
}

int main(int argc, char *argv[]) {
    
    CSR::CSRMAT<DATATYPE> *csrmat;
    
    int nloops = DEFAULT_NLOOPS;
    int nthreads = DEFAULT_NTHREADS;
    double precision = DEFAULT_PRECISION;

    CSR::ParseArgs<DATATYPE>(argc, argv, csrmat, nloops, nthreads, precision);

    std::cout << '\n' << "size:" << '\t' << csrmat->nrows << " x " << csrmat->ncols << '\n' << "nnz:" << '\t' << csrmat->nnzs << std::endl;
    std::cout << std::setprecision(3) << std::endl;

    // initialize vectors
    DATATYPE *x = (DATATYPE*)malloc(csrmat->ncols * sizeof(DATATYPE));
    DATATYPE *y1 = (DATATYPE*)malloc(csrmat->nrows * sizeof(DATATYPE));
    DATATYPE *y2 = (DATATYPE*)malloc(csrmat->nrows * sizeof(DATATYPE));
    for (int i = 0; i < csrmat->ncols; i++)
        x[i] = static_cast<DATATYPE>(1);
    for (int i = 0; i < csrmat->nrows; i++) {
        y1[i] = 0;
        y2[i] = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // CSR warmup
    for (int i = 0; i < 100; i++)
        CSR::SPMV_CSR<DATATYPE>(csrmat, x, y1, nthreads);
    
    // CSR test
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nloops; i++)
        CSR::SPMV_CSR<DATATYPE>(csrmat, x, y1, nthreads);
    auto end1 = std::chrono::high_resolution_clock::now();
    double time_spmv_csr = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    std::cout << "Time of CSR SPMV:" << '\t' << time_spmv_csr / nloops << " us" << '\t' << '\t' << csrmat->nnzs * 2 / (time_spmv_csr / nloops) / 1000 << " GFLOPS" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // GPU init
    DATATYPE *d_nnz_val, *d_x, *d_y;
    int *d_col_idx, *d_row_ptr;
    CHECK_CUDA( cudaMalloc((void**)&d_nnz_val, csrmat->nnzs * sizeof(DATATYPE)) );
    CHECK_CUDA( cudaMalloc((void**)&d_col_idx, csrmat->nnzs * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**)&d_row_ptr, csrmat->nrows * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**)&d_x, csrmat->ncols * sizeof(DATATYPE)) );
    CHECK_CUDA( cudaMalloc((void**)&d_y, csrmat->nrows * sizeof(DATATYPE)) );

    CHECK_CUDA( cudaMemcpy( d_nnz_val, csrmat->nnz_val, csrmat->nnzs * sizeof(DATATYPE), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( d_col_idx, csrmat->col_idx, csrmat->nnzs * sizeof(int), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( d_row_ptr, csrmat->row_ptr, csrmat->nrows * sizeof(int), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( d_x, x, csrmat->ncols * sizeof(DATATYPE), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( d_y, y2, csrmat->nrows * sizeof(DATATYPE), cudaMemcpyHostToDevice ) );

    dim3 blocksPerGrid(csrmat->nrows);  // blocks per grid
    dim3 threadsPerBlock(csrmat->ncols);  // threads per block
    double size_shared_mem = csrmat->ncols * sizeof(DATATYPE);  // size of shared memory of each block

    // GPU warmup
    for (int i = 0; i < 100; i++) {
        SPMV_GPU<DATATYPE><<<blocksPerGrid, threadsPerBlock, size_shared_mem>>>(d_nnz_val, d_col_idx, d_row_ptr, d_x, d_y);
        CHECK_CUDA( cudaDeviceSynchronize() );
    }
    
    // GPU test
    auto start2 = std::chrono::high_resolution_clock::now();  // CPU clock

    cudaEvent_t cudastart, cudastop;
    CHECK_CUDA( cudaEventCreate( &cudastart ) );
    CHECK_CUDA( cudaEventCreate( &cudastop ) );
    float cudatime = 0;
    for (int i = 0; i < nloops; i++) {
        CHECK_CUDA( cudaEventRecord(cudastart) );
        SPMV_GPU<DATATYPE><<<blocksPerGrid, threadsPerBlock, size_shared_mem>>>(d_nnz_val, d_col_idx, d_row_ptr, d_x, d_y);
        CHECK_CUDA( cudaEventRecord(cudastop) );
        CHECK_CUDA( cudaDeviceSynchronize() );
        
        float time;
        CHECK_CUDA( cudaEventElapsedTime(&time, cudastart, cudastop) );
        cudatime += time;
    }

    auto end2 = std::chrono::high_resolution_clock::now();  // CPU clock

    double time_spmv_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
    std::cout << "CPU_Time of GPU SPMV:" << '\t' << time_spmv_gpu / nloops << " us" << '\t' << '\t' << csrmat->nnzs * 2 / (time_spmv_gpu / nloops) / 1000 << " GFLOPS" << std::endl;
    std::cout << "GPU_Time of GPU SPMV:" << '\t' << cudatime / nloops << " us" << '\t' << '\t' << csrmat->nnzs * 2 / (cudatime / nloops) / 1000 << " GFLOPS" << std::endl;
    std::cout << "Speedup:" << '\t' << '\t' << time_spmv_csr / time_spmv_gpu << "x" << '\n' << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // verify results
    CSR::SPMV_CSR<DATATYPE>(csrmat, x, y1);
    SPMV_GPU<DATATYPE><<<blocksPerGrid, threadsPerBlock, size_shared_mem>>>(d_nnz_val, d_col_idx, d_row_ptr, d_x, d_y);
    CHECK_CUDA( cudaDeviceSynchronize() );

    // compare y1 and y2
    for (int i = 0; i < csrmat->nrows; i++)
        if (fabs(y1[i] - y2[i]) > precision && fabs(y1[i]-y2[i]) / fabs(y1[i]) > precision)
            std::cout << std::setprecision(20) << "Wrong result at row " << i << ": " << y1[i] << " vs " << y2[i] << std::endl;

    free(x);
    free(y1);
    free(y2);
    return 0;
}