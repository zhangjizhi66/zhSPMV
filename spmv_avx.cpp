
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ctime>
#include <chrono>
#include <immintrin.h>
#include "csr.h"

//#define VEC_LENGTH 256
//#define DATATYPE double
#define INDEXTYPE int32_t
#define SIMD_WIDTH (VEC_LENGTH / sizeof(DATATYPE) / 8)

// Parameters
#define DEFAULT_NLOOPS 1000
#define DEFAULT_NTHREADS 1
#define DEFAULT_PRECISION 1e-14
#define SIMD_USAGE_RATE 0.5  // lower limit, else apply in-row simd
#define LONG_ROW_LENGTH 100  // lower limit, else apply in-row simd

// Optional Function
#define INROW_SIMD

template <typename T, typename U>
class SIMDMAT {
public:
    SIMDMAT(int csr_nrows, int *csr_row_ptr, U *csr_col_idx, T *csr_nnz_val);
    SIMDMAT(const SIMDMAT &) = delete;  // 禁止拷贝构造函数
    SIMDMAT &operator=(const SIMDMAT &) = delete;  // 禁止拷贝赋值函数
    SIMDMAT(SIMDMAT&&) = default;
    SIMDMAT& operator=(SIMDMAT&&) = default;
    ~SIMDMAT() {
        free(nnz_val);
        free(col_idx);
        free(simd_ptr);
        free(sort_raw);
    }

    int nrows;
    int nsimdblocks_crossrow;
    int nsimdblocks_inrow;
    int nrows_crossrow_simd;
    int nrows_inrow_simd;

    T *nnz_val;
    U *col_idx;
    int *simd_ptr;  // pointer of each simd block

    int *sort_raw;  // sorted index to raw row index
};

template <typename T, typename U>
SIMDMAT<T, U>::SIMDMAT(int csr_nrows, int *csr_row_ptr, U *csr_col_idx, T *csr_nnz_val) {
    nrows = csr_nrows;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // row sorting
    struct ROW {
        int rawidx;
        int first_col_idx;
        int length;
    };

    std::vector<ROW> rows(nrows);
    rows.reserve(2*nrows);
    for (int i = 0; i < nrows; i++) {
        rows[i].rawidx = i;
        rows[i].first_col_idx = csr_col_idx[csr_row_ptr[i]];
        rows[i].length = csr_row_ptr[i + 1] - csr_row_ptr[i];
    }
    std::sort(rows.begin(), rows.end(), [](const ROW &a, const ROW &b) { 
        if ( a.length == b.length )
            return a.first_col_idx < b.first_col_idx;
        return a.length > b.length;
    });

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // find in-row simd rows
    std::vector<int> in_row_simd;  // sorted row indexes of in-row simd rows
    in_row_simd.reserve(nrows);
    std::vector<ROW> in_row_simd_data;
    in_row_simd_data.reserve(nrows);
    for (int i = 0; i < nrows; ) {
        // judge if the simd block is feasible
        bool simd_verify = true;
        int nnzs_block = 0;
        for (int j = 0; j < int(SIMD_WIDTH); j++) {
            if ( i + j >= nrows )
                break;
            else
                nnzs_block += rows[i + j].length;
        }
        float simd_usage_rate = (float)nnzs_block / (rows[i].length * SIMD_WIDTH);
        if ( simd_usage_rate < SIMD_USAGE_RATE && rows[i].length > LONG_ROW_LENGTH )
            simd_verify = false;

        if ( !simd_verify ) {
            in_row_simd.push_back(i);
            in_row_simd_data.push_back(rows[i]);
            std::cout << "Extremely long row:" << '\t' 
                      << "index = " << rows[i].rawidx << '\t' 
                      << "length = " << rows[i].length << '\n';
            i++;
        } else
            i += SIMD_WIDTH;
    }
    in_row_simd.push_back(nrows);

    if ( in_row_simd.size() > 1 )
#ifdef INROW_SIMD
        std::cout << '\n' << "Using in-row SIMD for extremely long rows ..." << '\n' << std::endl;
#else
        std::cout << '\n' << "Using scalar operation for extremely long rows ..." << '\n' << std::endl;
#endif

    nrows_inrow_simd = in_row_simd.size() - 1;
    nrows_crossrow_simd = nrows - nrows_inrow_simd;
    for (int j = 0; j < nrows_inrow_simd; j++) {
        for (int i = in_row_simd[j] + 1; i < in_row_simd[j+1]; i++) {
            rows[i - j - 1].rawidx = rows[i].rawidx;
            rows[i - j - 1].first_col_idx = rows[i].first_col_idx;
            rows[i - j - 1].length = rows[i].length;
        }
    }
    rows.resize(nrows - nrows_inrow_simd);
    rows.insert(rows.end(), in_row_simd_data.begin(), in_row_simd_data.end());

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    sort_raw = (int*)malloc(nrows * sizeof(int));
    for (int i = 0; i < nrows; i++)
        sort_raw[i] = rows[i].rawidx;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int Nsimd = 0;
    for (int i = 0; i < nrows - nrows_inrow_simd; i += SIMD_WIDTH)
        Nsimd += rows[i].length;
    for (int i = nrows - nrows_inrow_simd; i < nrows; i++)
        Nsimd += std::ceil((double)rows[i].length / SIMD_WIDTH);

    nsimdblocks_crossrow = std::ceil((double)(nrows - nrows_inrow_simd) / SIMD_WIDTH);
    nsimdblocks_inrow = nrows_inrow_simd;

    posix_memalign((void**)&nnz_val, VEC_LENGTH / 8, Nsimd * SIMD_WIDTH * sizeof(T));
    posix_memalign((void**)&col_idx, VEC_LENGTH / 8, Nsimd * SIMD_WIDTH * sizeof(U));
    simd_ptr = (int*)malloc((nsimdblocks_crossrow + nsimdblocks_inrow + 1) * sizeof(int));

    int simd_ptr_idx = 0, isimd = 0;
    simd_ptr[isimd++] = 0;

    // cross-row simd blocks
    for (int irow = 0; irow < int(nrows - nrows_inrow_simd - SIMD_WIDTH); irow += SIMD_WIDTH) {
        int max_idx = csr_row_ptr[sort_raw[irow]+1] - csr_row_ptr[sort_raw[irow]];
        for (int idx = 0; idx < max_idx; idx++) {
            for (int i = 0; i < int(SIMD_WIDTH); i++) {
                int raw_row_idx = sort_raw[irow + i];
                if ( csr_row_ptr[raw_row_idx + 1] - csr_row_ptr[raw_row_idx] <= idx ) {
                    nnz_val[simd_ptr_idx] = 0;
                    col_idx[simd_ptr_idx] = csr_col_idx[csr_row_ptr[sort_raw[irow]] + idx];
                } else {
                    int idx_temp = csr_row_ptr[raw_row_idx] + idx;
                    nnz_val[simd_ptr_idx] = csr_nnz_val[idx_temp];
                    col_idx[simd_ptr_idx] = csr_col_idx[idx_temp];
                }
                simd_ptr_idx++;
            }
        }
        simd_ptr[isimd++] = simd_ptr_idx;
    }

    // the last cross-row simd block
    {
        int irow = SIMD_WIDTH * (isimd - 1);
        int max_idx = csr_row_ptr[sort_raw[irow] + 1] - csr_row_ptr[sort_raw[irow]];
        for (int idx = 0; idx < max_idx; idx++) {
            for (int i = 0; i < int(SIMD_WIDTH); i++) {
                int sort_row_idx = irow + i;
                int raw_row_idx = sort_raw[sort_row_idx];
                if ( sort_row_idx >= nrows - nrows_inrow_simd ) {  // use col_idx in the first row in the simd block
                    nnz_val[simd_ptr_idx] = 0;
                    col_idx[simd_ptr_idx] = csr_col_idx[csr_row_ptr[sort_raw[irow]] + idx];
                } else if ( csr_row_ptr[raw_row_idx + 1] - csr_row_ptr[raw_row_idx] <= idx ) {
                    nnz_val[simd_ptr_idx] = 0;
                    col_idx[simd_ptr_idx] = csr_col_idx[csr_row_ptr[sort_raw[irow]] + idx];
                } else {
                    int idx_temp = csr_row_ptr[raw_row_idx] + idx;
                    nnz_val[simd_ptr_idx] = csr_nnz_val[idx_temp];
                    col_idx[simd_ptr_idx] = csr_col_idx[idx_temp];
                }
                simd_ptr_idx++;
            }
        }
        simd_ptr[isimd++] = simd_ptr_idx;
    }

    // in-row simd blocks
    for (int irow = nrows - nrows_inrow_simd; irow < nrows; irow++) {
        int max_idx = csr_row_ptr[sort_raw[irow] + 1] - csr_row_ptr[sort_raw[irow]];
        int start_idx = csr_row_ptr[sort_raw[irow]];
        for (int idx = 0; idx < max_idx; idx++) {
            int this_idx = start_idx + idx;
            nnz_val[simd_ptr_idx] = csr_nnz_val[this_idx];
            col_idx[simd_ptr_idx] = csr_col_idx[this_idx];
            simd_ptr_idx++;
        }
        if ( max_idx % SIMD_WIDTH != 0 ) {
            int end_idx = start_idx + max_idx;
            int nmasks = SIMD_WIDTH - max_idx % SIMD_WIDTH;
            for (int i = 0; i < nmasks; i++) {
                nnz_val[simd_ptr_idx] = 0;
                col_idx[simd_ptr_idx] = csr_col_idx[end_idx];
                simd_ptr_idx++;
            }
        }
        simd_ptr[isimd++] = simd_ptr_idx;
    }
}

template <typename T, typename U>
void SPMV_SIMD256d(SIMDMAT<T, U> &simdmat, T *x, T *y, int nthreads = 1) {
    int start_row = 0;

    // cross-row simd blocks
    int end_simd = simdmat.nsimdblocks_crossrow - 1;
    for (int isimd = 0; isimd < end_simd; isimd++) {
        __m256d y_vec = _mm256_setzero_pd();
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
        for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
            __m128i idx_vec = _mm_load_si128((__m128i *)&simdmat.col_idx[idx]);
            __m256d x_vec = _mm256_i32gather_pd(x, idx_vec, sizeof(DATATYPE));
            __m256d val_vec = _mm256_load_pd(&simdmat.nnz_val[idx]);
            y_vec = _mm256_fmadd_pd(val_vec, x_vec, y_vec);
        }
        __m128i row_idx_vec = _mm_loadu_si128((__m128i *)&simdmat.sort_raw[start_row]);
        _mm256_i32scatter_pd(y, row_idx_vec, y_vec, sizeof(DATATYPE));
        start_row += SIMD_WIDTH;
    }
    
    // the last cross-row simd block
    {
        int isimd = end_simd;  // simdmat.nsimdblocks_crossrow - 1
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
        __m256d y_vec = _mm256_setzero_pd();
        for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
            __m128i idx_vec = _mm_load_si128((__m128i *)&simdmat.col_idx[idx]);
            __m256d x_vec = _mm256_i32gather_pd(x, idx_vec, sizeof(DATATYPE));
            __m256d val_vec = _mm256_load_pd(&simdmat.nnz_val[idx]);
            y_vec = _mm256_fmadd_pd(val_vec, x_vec, y_vec);
        }
        if ( simdmat.nrows_crossrow_simd % SIMD_WIDTH != 0 ) {
            double y_arr[SIMD_WIDTH];
            _mm256_storeu_pd(y_arr, y_vec);
            int end_i = simdmat.nrows_crossrow_simd % SIMD_WIDTH;
            for (int i = 0; i < end_i; i++)
                y[simdmat.sort_raw[isimd * SIMD_WIDTH + i]] = y_arr[i];
        } else {
            __m128i row_idx_vec = _mm_loadu_si128((__m128i *)&simdmat.sort_raw[isimd * SIMD_WIDTH]);
            _mm256_i32scatter_pd(y, row_idx_vec, y_vec, sizeof(DATATYPE));
        }
    }

    // in-row simd blocks
    int start_row_idx_inrow = simdmat.nrows - simdmat.nrows_inrow_simd;
    int nsimdblocks = simdmat.nsimdblocks_crossrow + simdmat.nsimdblocks_inrow;
    for (int isimd = simdmat.nsimdblocks_crossrow; isimd < nsimdblocks; isimd++) {
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
#ifdef INROW_SIMD
        __m256d y_vec = _mm256_setzero_pd();
        for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
            __m128i idx_vec = _mm_load_si128((__m128i *)&simdmat.col_idx[idx]);
            __m256d x_vec = _mm256_i32gather_pd(x, idx_vec, sizeof(DATATYPE));
            __m256d val_vec = _mm256_load_pd(&simdmat.nnz_val[idx]);
            y_vec = _mm256_fmadd_pd(val_vec, x_vec, y_vec);
        }
        DATATYPE y_arr[SIMD_WIDTH];
        _mm256_storeu_pd(y_arr, y_vec);
        y[simdmat.sort_raw[start_row_idx_inrow + isimd - simdmat.nsimdblocks_crossrow]] = std::accumulate(y_arr, y_arr + SIMD_WIDTH, 0.0);
#else
        int raw_row_idx = simdmat.sort_raw[start_row_idx_inrow + isimd - simdmat.nsimdblocks_crossrow];
        y[raw_row_idx] = 0;
        for (int idx = start_idx; idx < end_idx; idx++)
            y[raw_row_idx] += simdmat.nnz_val[idx] * x[simdmat.col_idx[idx]];
#endif
    }
}

template <typename T, typename U>
void SPMV_SIMD512d(SIMDMAT<T, U> &simdmat, T *x, T *y, int nthreads = 1) {
    int start_row = 0;

    // cross-row simd blocks
    int end_simd = simdmat.nsimdblocks_crossrow - 1;
    for (int isimd = 0; isimd < end_simd; isimd++) {
        __m512d y_vec = _mm512_setzero_pd();
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
        for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
            __m256i idx_vec = _mm256_load_si256((__m256i *)&simdmat.col_idx[idx]);
            __m512d x_vec = _mm512_i32gather_pd(idx_vec, x, sizeof(T));
            __m512d val_vec = _mm512_load_pd(&simdmat.nnz_val[idx]);
            y_vec = _mm512_fmadd_pd(val_vec, x_vec, y_vec);
        }
        __m256i row_idx_vec = _mm256_loadu_si256((__m256i *)&simdmat.sort_raw[start_row]);
        _mm512_i32scatter_pd(y, row_idx_vec, y_vec, sizeof(T));
        start_row += SIMD_WIDTH;
    }

    // the last cross-row simd block
    {
        int isimd = end_simd;  // simdmat.nsimdblocks_crossrow - 1
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
        __m512d y_vec = _mm512_setzero_pd();
        for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
            __m256i idx_vec = _mm256_load_si256((__m256i *)&simdmat.col_idx[idx]);
            __m512d x_vec = _mm512_i32gather_pd(idx_vec, x, sizeof(T));
            __m512d val_vec = _mm512_load_pd(&simdmat.nnz_val[idx]);
            y_vec = _mm512_fmadd_pd(val_vec, x_vec, y_vec);
        }
        if ( simdmat.nrows_crossrow_simd % SIMD_WIDTH != 0 ) {
            double y_arr[SIMD_WIDTH];
            _mm512_storeu_pd(y_arr, y_vec);
            int end_i = simdmat.nrows_crossrow_simd % SIMD_WIDTH;
            for (int i = 0; i < end_i; i++)
                y[simdmat.sort_raw[isimd * SIMD_WIDTH + i]] = y_arr[i];
        } else {
            __m256i row_idx_vec = _mm256_loadu_si256((__m256i *)&simdmat.sort_raw[isimd * SIMD_WIDTH]);
            _mm512_i32scatter_pd(y, row_idx_vec, y_vec, sizeof(T));
        }
    }

    // in-row simd blocks
    int start_row_idx_inrow = simdmat.nrows - simdmat.nrows_inrow_simd;
    int nsimdblocks = simdmat.nsimdblocks_crossrow + simdmat.nsimdblocks_inrow;
    for (int isimd = simdmat.nsimdblocks_crossrow; isimd < nsimdblocks; isimd++) {
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
#ifdef INROW_SIMD
        __m512d y_vec = _mm512_setzero_pd();
        for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
            __m256i idx_vec = _mm256_load_si256((__m256i *)&simdmat.col_idx[idx]);
            __m512d x_vec = _mm512_i32gather_pd(idx_vec, x, sizeof(T));
            __m512d val_vec = _mm512_load_pd(&simdmat.nnz_val[idx]);
            y_vec = _mm512_fmadd_pd(val_vec, x_vec, y_vec);
        }
        y[simdmat.sort_raw[start_row_idx_inrow + isimd - simdmat.nsimdblocks_crossrow]] = _mm512_reduce_add_pd(y_vec);
#else
        int raw_row_idx = simdmat.sort_raw[start_row_idx_inrow + isimd - simdmat.nsimdblocks_crossrow];
        y[raw_row_idx] = 0;
        for (int idx = start_idx; idx < end_idx; idx++)
            y[raw_row_idx] += simdmat.nnz_val[idx] * x[simdmat.col_idx[idx]];
#endif
    }
}

int main(int argc, char *argv[]) {
    
    CSR::CSRMAT<DATATYPE> *csrmat;
    
    int nloops = DEFAULT_NLOOPS;
    int nthreads = DEFAULT_NTHREADS;
    double precision = DEFAULT_PRECISION;

    CSR::ParseArgs<DATATYPE>(argc, argv, csrmat, nloops, nthreads, precision);
    
    std::cout << '\n' << "size:" << '\t' << csrmat->nrows << " x " << csrmat->ncols << '\n' << "nnz:" << '\t' << csrmat->nnzs << std::endl;
    std::cout << std::setprecision(3) << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // preprocess matrix
    auto start = std::chrono::high_resolution_clock::now();

    SIMDMAT<DATATYPE, INDEXTYPE> simdmat(csrmat->nrows, csrmat->row_ptr, csrmat->col_idx, csrmat->nnz_val);

    auto end = std::chrono::high_resolution_clock::now();
    double time_preprocess = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time of preprocessing:" << '\t' << time_preprocess << " us" << '\n' << std::endl;

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

    // SIMD warmup
    for (int i = 0; i < 100; i++)
#if VEC_LENGTH == 256 && DATATYPE == double
        SPMV_SIMD256d<DATATYPE>(simdmat, x, y2, nthreads);
#elif VEC_LENGTH == 512 && DATATYPE == double
        SPMV_SIMD512d<DATATYPE>(simdmat, x, y2, nthreads);
#endif
    
    // SIMD test
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nloops; i++)
#if VEC_LENGTH == 256 && DATATYPE == double
        SPMV_SIMD256d<DATATYPE>(simdmat, x, y2, nthreads);
#elif VEC_LENGTH == 512 && DATATYPE == double
        SPMV_SIMD512d<DATATYPE>(simdmat, x, y2, nthreads);
#endif
    auto end2 = std::chrono::high_resolution_clock::now();
    double time_spmv_simd = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
    std::cout << "Time of SIMD SPMV:" << '\t' << time_spmv_simd / nloops << " us" << '\t' << '\t' << csrmat->nnzs * 2 / (time_spmv_simd / nloops) / 1000 << " GFLOPS" << std::endl;
    std::cout << "Speedup:" << '\t' << '\t' << time_spmv_csr / time_spmv_simd << "x" << '\n' << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // verify results
    CSR::SPMV_CSR<DATATYPE>(csrmat, x, y1);
#if VEC_LENGTH == 256 && DATATYPE == double
    SPMV_SIMD256d<DATATYPE>(simdmat, x, y2);
#elif VEC_LENGTH == 512 && DATATYPE == double
    SPMV_SIMD512d<DATATYPE>(simdmat, x, y2);
#endif

    // compare y1 and y2
    for (int i = 0; i < csrmat->nrows; i++)
        if (fabs(y1[i] - y2[i]) > precision && fabs(y1[i]-y2[i]) / fabs(y1[i]) > precision)
            std::cout << std::setprecision(20) << "Wrong result at row " << i << ": " << y1[i] << " vs " << y2[i] << std::endl;

    free(x);
    free(y1);
    free(y2);
    return 0;
}