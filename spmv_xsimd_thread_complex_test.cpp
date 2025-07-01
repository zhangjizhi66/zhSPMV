
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include "xsimd/xsimd.hpp"
#include "new_gather_scatter.hpp"
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
#define THREAD_TIME_TEST
#define CSR_TIME_TEST

template <typename T>
int Cost(const T* col_idx, int size) {
    return size + 0.1;
}

template <typename T, typename U>
class SIMDMAT {
public:
    SIMDMAT(int csr_nrows, int *csr_row_ptr, U *csr_col_idx, T *csr_nnz_val, const int nthreads);
    SIMDMAT(const SIMDMAT &) = delete;  // 禁止拷贝构造函数
    SIMDMAT &operator=(const SIMDMAT &) = delete;  // 禁止拷贝赋值函数
    SIMDMAT(SIMDMAT&&) = default;
    SIMDMAT& operator=(SIMDMAT&&) = default;
    ~SIMDMAT() {
        free(nnz_val);
        free(col_idx);
        free(simd_ptr);
        free(sort_raw);
        free(sort_raw_complex);
    }

    int nrows;
    int nnzs;
    int cost;

    int nrows_crossrow_simd;
    int nrows_scalar;
    int nrows_inrow_simd;

    T *nnz_val;
    U *col_idx;
    int *simd_ptr;  // pointer of each simd block

    int *sort_raw;  // sorted index to raw row index
    int *sort_raw_complex;
};

template <typename T, typename U>
SIMDMAT<T, U>::SIMDMAT(int csr_nrows, int *csr_row_ptr, U *csr_col_idx, T *csr_nnz_val, const int nthreads) {
    nrows = csr_nrows;
    nnzs = csr_row_ptr[csr_nrows];
    cost = 0;
    for (int i = 0; i < nrows; i++) {
        int row_length = csr_row_ptr[i + 1] - csr_row_ptr[i];
        cost += Cost<U>(&csr_col_idx[csr_row_ptr[i]], row_length);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // row sorting
    struct ROW {
        int rawidx;
        int first_col_idx;
        int length;
        ROW(int a = -1, int b = -1, int c = 0) : rawidx(a), first_col_idx(b), length(c) {}
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
        for (int j = 0; j < int(SIMD_WIDTH / 2); j++) {
            if ( i + j >= nrows )
                break;
            else
                nnzs_block += rows[i + j].length * 2;
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
            i += SIMD_WIDTH / 2;
    }
    in_row_simd.push_back(nrows);

    nrows_inrow_simd = in_row_simd.size() - 1;
    nrows_scalar = ( nrows - nrows_inrow_simd ) % SIMD_WIDTH;
    nrows_crossrow_simd = nrows - nrows_inrow_simd - nrows_scalar;
    for (int j = 0; j < nrows_inrow_simd; j++) {
        for (int i = in_row_simd[j] + 1; i < in_row_simd[j + 1]; i++) {
            rows[i - j - 1].rawidx = rows[i].rawidx;
            rows[i - j - 1].first_col_idx = rows[i].first_col_idx;
            rows[i - j - 1].length = rows[i].length;
        }
    }
    rows.resize(nrows - nrows_inrow_simd);
    rows.insert(rows.end(), in_row_simd_data.begin(), in_row_simd_data.end());

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    sort_raw = (int*)malloc(nrows * sizeof(int));
    sort_raw_complex = (int*)malloc(2 * nrows * sizeof(int));
    for (int i = 0; i < nrows; i++) {
        sort_raw[i] = rows[i].rawidx;
        sort_raw_complex[2 * i] = rows[i].rawidx * 2;
        sort_raw_complex[2 * i + 1] = rows[i].rawidx * 2 + 1;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int N = 0;
    for (int i = 0; i < nrows_crossrow_simd; i += SIMD_WIDTH / 2)
        N += rows[i].length;
    for (int i = nrows_crossrow_simd + nrows_scalar; i < nrows; i++)
        N += std::ceil((double)rows[i].length / SIMD_WIDTH) * 2;
    N *= SIMD_WIDTH;
    for (int i = nrows_crossrow_simd; i < nrows_crossrow_simd + nrows_scalar; i++)
        N += rows[i].length * 2;

    posix_memalign((void**)&nnz_val, VEC_LENGTH / 8, N * sizeof(T));
    posix_memalign((void**)&col_idx, VEC_LENGTH / 8, N * sizeof(U));
    simd_ptr = (int*)malloc(((nrows_crossrow_simd / SIMD_WIDTH + nrows_scalar + nrows_inrow_simd) * 2 + 1) * sizeof(int));

    int simd_ptr_idx = 0, isimd = 0;
    simd_ptr[isimd++] = 0;

    // cross-row simd blocks
    for (int irow = 0; irow < nrows_crossrow_simd; irow += SIMD_WIDTH / 2) {
        int max_idx = csr_row_ptr[sort_raw[irow] + 1] - csr_row_ptr[sort_raw[irow]];
        for (int idx = 0; idx < max_idx; idx++) {
            for (int i = 0; i < int(SIMD_WIDTH / 2); i++) {
                int raw_row_idx = sort_raw[irow + i];
                if ( csr_row_ptr[raw_row_idx + 1] - csr_row_ptr[raw_row_idx] <= idx ) {
                    nnz_val[simd_ptr_idx] = 0;
                    col_idx[simd_ptr_idx] = csr_col_idx[csr_row_ptr[sort_raw[irow]] + idx];
                    simd_ptr_idx++;
                    nnz_val[simd_ptr_idx] = 0;
                    col_idx[simd_ptr_idx] = csr_col_idx[csr_row_ptr[sort_raw[irow]] + idx];
                } else {
                    int idx_temp = csr_row_ptr[raw_row_idx] + idx;
                    nnz_val[simd_ptr_idx] = csr_nnz_val[idx_temp];
                    col_idx[simd_ptr_idx] = csr_col_idx[idx_temp];
                    simd_ptr_idx++;
                    nnz_val[simd_ptr_idx] = csr_nnz_val[idx_temp];
                    col_idx[simd_ptr_idx] = csr_col_idx[idx_temp];
                }
                simd_ptr_idx++;
            }
        }
        simd_ptr[isimd++] = simd_ptr_idx;
    }

    // scalar rows & in-row simd rows
    for (int irow = nrows_crossrow_simd; irow < nrows; irow++) {
        int max_idx = csr_row_ptr[sort_raw[irow] + 1] - csr_row_ptr[sort_raw[irow]];
        int start_idx = csr_row_ptr[sort_raw[irow]];

        // conductance
        for (int idx = 0; idx < max_idx; idx++) {
            int this_idx = start_idx + idx;
            nnz_val[simd_ptr_idx] = csr_nnz_val[this_idx];
            col_idx[simd_ptr_idx] = csr_col_idx[this_idx];
            simd_ptr_idx++;
        }
#ifdef INROW_SIMD
        if ( max_idx % SIMD_WIDTH != 0 && irow >= nrows_crossrow_simd + nrows_scalar ) {
            int end_idx = start_idx + max_idx;
            int nmasks = SIMD_WIDTH - max_idx % SIMD_WIDTH;
            for (int i = 0; i < nmasks; i++) {
                nnz_val[simd_ptr_idx] = 0;
                col_idx[simd_ptr_idx] = csr_col_idx[end_idx];
                simd_ptr_idx++;
            }
        }
#endif
        simd_ptr[isimd++] = simd_ptr_idx;

        // compacitance
        for (int idx = 0; idx < max_idx; idx++) {
            int this_idx = start_idx + idx;
            nnz_val[simd_ptr_idx] = csr_nnz_val[this_idx];
            col_idx[simd_ptr_idx] = csr_col_idx[this_idx];
            simd_ptr_idx++;
        }
#ifdef INROW_SIMD
        if ( max_idx % SIMD_WIDTH != 0 && irow >= nrows_crossrow_simd + nrows_scalar ) {
            int end_idx = start_idx + max_idx;
            int nmasks = SIMD_WIDTH - max_idx % SIMD_WIDTH;
            for (int i = 0; i < nmasks; i++) {
                nnz_val[simd_ptr_idx] = 0;
                col_idx[simd_ptr_idx] = csr_col_idx[end_idx];
                simd_ptr_idx++;
            }
        }
#endif
        simd_ptr[isimd++] = simd_ptr_idx;
    }
}

template <typename T, typename U, class Arch>
void SPMV_SIMD_thread(SIMDMAT<T, U> &simdmat, T *x, T *y) {
    using VAL_VEC_TYPE = xsimd::batch<T, Arch>;
    using IDX_VEC_TYPE = xsimd::batch<U, Arch>;

    // cross-row simd blocks
    int start_simd = 0;
    int end_simd = simdmat.nrows_crossrow_simd * 2 / SIMD_WIDTH;
    for (int isimd = start_simd; isimd < end_simd; isimd++) {
        VAL_VEC_TYPE y_vec(0);
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
        for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
            IDX_VEC_TYPE idx_vec = IDX_VEC_TYPE::load_unaligned(&simdmat.col_idx[idx]);
            VAL_VEC_TYPE x_vec = VAL_VEC_TYPE::gather(x, idx_vec);
            VAL_VEC_TYPE val_vec = VAL_VEC_TYPE::load_unaligned(&simdmat.nnz_val[idx]);
            y_vec = xsimd::fma(val_vec, x_vec, y_vec);
        }
        IDX_VEC_TYPE row_idx_vec = IDX_VEC_TYPE::load_unaligned(&simdmat.sort_raw_complex[isimd * SIMD_WIDTH]);
        y_vec.scatter(y, row_idx_vec);
    }

    // scalar rows & in-row simd rows
    int nsimdblocks1 = simdmat.nrows_crossrow_simd * 2 / SIMD_WIDTH;
    int nsimdblocks2 = nsimdblocks1 + simdmat.nrows_scalar * 2;
    int nsimdblocks3 = nsimdblocks2 + simdmat.nrows_inrow_simd * 2;
    for (int isimd = nsimdblocks1; isimd < nsimdblocks3; isimd++) {
        int start_idx = simdmat.simd_ptr[isimd];
        int end_idx = simdmat.simd_ptr[isimd + 1];
#ifdef INROW_SIMD
        if ( isimd >= nsimdblocks2 ) {
            VAL_VEC_TYPE y_vec(0);
            for (int idx = start_idx; idx < end_idx; idx += SIMD_WIDTH) {
                IDX_VEC_TYPE idx_vec = IDX_VEC_TYPE::load_unaligned(&simdmat.col_idx[idx]);
                VAL_VEC_TYPE x_vec = VAL_VEC_TYPE::gather(x, idx_vec);
                VAL_VEC_TYPE val_vec = VAL_VEC_TYPE::load_unaligned(&simdmat.nnz_val[idx]);
                y_vec = xsimd::fma(val_vec, x_vec, y_vec);
            }
            y[simdmat.sort_raw_complex[simdmat.nrows_crossrow_simd * 2 + isimd - nsimdblocks1]] = xsimd::reduce_add(y_vec);
        }
        else
#endif
        {
            int raw_row_idx = simdmat.sort_raw_complex[simdmat.nrows_crossrow_simd * 2 + isimd - nsimdblocks1];
            y[raw_row_idx] = 0;
            for (int idx = start_idx; idx < end_idx; idx++)
                y[raw_row_idx] += simdmat.nnz_val[idx] * x[simdmat.col_idx[idx]];
        }
    }
}

template <typename T, typename U, class Arch>
void SPMV_SIMD(std::vector<SIMDMAT<T, U>> &vsimdmat, std::vector<int> &vstartrow, std::vector<double> &vtimethread, T *x, T *y, int nthreads = 1) {
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int ithread = 0; ithread < nthreads; ithread++) {
#ifdef THREAD_TIME_TEST
        auto start = std::chrono::high_resolution_clock::now();
#endif

        SPMV_SIMD_thread<T, U, Arch>(vsimdmat[ithread], x, &y[vstartrow[ithread]]);

#ifdef THREAD_TIME_TEST
        auto end = std::chrono::high_resolution_clock::now();
        double time_thread = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        vtimethread[ithread] += time_thread;
#endif
    }
}

// 判断是否可以将数组划分为最多m个区间，使得每个区间和不超过maxSum
bool canPartition(const std::vector<int>& nums, int m, int maxSum) {
    int partitions = 1;  // 至少需要1个区间
    int currentSum = 0;
    
    for (int num : nums) {
        if ( num > maxSum ) return false;
        
        if ( currentSum + num <= maxSum )  // 尝试将当前元素加入当前区间
            currentSum += num;
        else {  // 不能加入当前区间，需要新的区间
            partitions++;
            if ( partitions > m ) return false;  // 需要的区间数超过m
            currentSum = num;
        }
    }
    return true;
}

// 将vector划分为n个区间，使最大区间和最小
int minMaxSubarraySum(std::vector<int>& nums, int n) {
    if ( nums.empty() || n <= 0 || n > nums.size() ) {
        std::cerr << "Invalid input" << std::endl;
        return 0;
    }
    
    int left = *std::max_element(nums.begin(), nums.end());  // 二分查找的下界：数组中的最大值
    int right = std::accumulate(nums.begin(), nums.end(), 0.0);  // 二分查找的上界：数组所有元素的和
    
    // 二分查找最小的可行maxSum
    while ( left + 1 < right ) {
        int mid = left + (right - left) / 2;
        if ( canPartition(nums, n, mid) ) right = mid;
        else left = mid;
    }

    if ( canPartition(nums, n, left) ) return left;
    else if ( canPartition(nums, n, right) ) return right;
    else {
        std::cerr << "No valid partition found" << std::endl;
        return 0;
    }
}

template <typename T, typename U>
std::vector<SIMDMAT<T, U>> thread_partition(int csr_nrows, int *csr_row_ptr, U *csr_col_idx, T *csr_nnz_val, int nthreads = 1) {
    std::vector<int> row_costs(csr_nrows, 0.0);
    for (int i = 0; i < csr_nrows; i++) {
        int row_length = csr_row_ptr[i + 1] - csr_row_ptr[i];
        row_costs[i] = Cost<U>(csr_col_idx + csr_row_ptr[i], row_length);
        //std::cout << "Row " << i << ": length = " << row_length << ", cost = " << row_costs[i] << std::endl;
    }

    int max_cost = minMaxSubarraySum(row_costs, nthreads);
    //std::cout << "Max cost per row: " << max_cost << '\n' << std::endl;

    std::vector<SIMDMAT<T, U>> vsimdmat;
    vsimdmat.reserve(nthreads);

    int start_row = 0, end_row = 0;
    for (int ithread = 0; ithread < nthreads; ithread++) {
        int thread_cost = 0;
        if ( start_row < csr_nrows ) {
            while ( true ) {
                thread_cost += row_costs[end_row];
                if ( end_row + 1 >= csr_nrows )
                    break;
                if ( thread_cost + row_costs[end_row + 1] > max_cost )
                    break;
                end_row++;
            }
            end_row++;

            if ( ithread == nthreads - 1 )
                end_row = csr_nrows;
        }

        //std::cout << "Thread " << ithread << ": start_row = " << start_row << ", end_row = " << end_row << ", thread_cost = " << thread_cost << std::endl;

        int nrows = end_row - start_row;
        std::vector<int> row_ptr(nrows + 1);
        for (int i = 0; i < nrows + 1; i++)
            row_ptr[i] = csr_row_ptr[start_row + i] - csr_row_ptr[start_row];

        vsimdmat.emplace_back(nrows, row_ptr.data(), &csr_col_idx[csr_row_ptr[start_row]], &csr_nnz_val[csr_row_ptr[start_row]], nthreads);

        start_row = end_row;
    }
    return vsimdmat;
}

template <typename T>
void SPMV_CSR_complex(CSR::CSRMAT<T> *csrmat, T *x, T *y, int nthreads = 1) {
    int nrows_per_thread = (csrmat->nrows + nthreads - 1) / nthreads;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int ithread = 0; ithread < nthreads; ithread++) {
        int start_row = ithread * nrows_per_thread;
        int end_row = std::min(start_row + nrows_per_thread, csrmat->nrows);
        for (int i = start_row; i < end_row; i++) {
            double sum1 = 0;
            for (int j = csrmat->row_ptr[i]; j < csrmat->row_ptr[i + 1]; j++)
                sum1 += csrmat->nnz_val[j] * x[csrmat->col_idx[j]];
            y[2 * i] = sum1;

            double sum2 = 0;
            for (int j = csrmat->row_ptr[i]; j < csrmat->row_ptr[i + 1]; j++)
                sum2 += csrmat->nnz_val[j] * x[csrmat->col_idx[j]];
            y[2 * i + 1] = sum2;
        }
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

    std::vector<SIMDMAT<DATATYPE, INDEXTYPE>> vsimdmat = thread_partition<DATATYPE, INDEXTYPE>(csrmat->nrows, csrmat->row_ptr, csrmat->col_idx, csrmat->nnz_val, nthreads);
    std::vector<int> vstartrow(nthreads);
    for (int i = 0; i < nthreads; i++) {
        vstartrow[i] = (i == 0) ? 0 : vstartrow[i - 1] + vsimdmat[i - 1].nrows * 2;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_preprocess = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << '\n' << "Time of preprocessing:" << '\t' << time_preprocess << " us" << '\n' << std::endl;

    // initialize vectors
    DATATYPE *x = (DATATYPE*)malloc(csrmat->ncols * sizeof(DATATYPE));
    DATATYPE *y1 = (DATATYPE*)malloc(csrmat->nrows * 2 * sizeof(DATATYPE));
    DATATYPE *y2 = (DATATYPE*)malloc(csrmat->nrows * 2 * sizeof(DATATYPE));
    //DATATYPE *y2;
    //posix_memalign((void**)&y2, sysconf(_SC_LEVEL1_DCACHE_LINESIZE), csrmat->nrows * sizeof(DATATYPE));
    for (int i = 0; i < csrmat->ncols; i++)
        x[i] = static_cast<DATATYPE>(1);
    for (int i = 0; i < csrmat->nrows * 2; i++) {
        y1[i] = 0;
        y2[i] = 0;
    }

#ifdef INROW_SIMD
    std::cout << "Using in-row SIMD for extremely long rows ..." << '\n' << std::endl;
#else
    std::cout << "Using scalar operation for extremely long rows ..." << '\n' << std::endl;
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CSR_TIME_TEST
    // CSR warmup
    for (int i = 0; i < 100; i++)
        SPMV_CSR_complex<DATATYPE>(csrmat, x, y1, nthreads);
    
    // CSR test
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nloops; i++)
        SPMV_CSR_complex<DATATYPE>(csrmat, x, y1, nthreads);
    auto end1 = std::chrono::high_resolution_clock::now();
    double time_spmv_csr = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    std::cout << "Time of CSR SPMV:" << '\t' << time_spmv_csr / nloops << " us" << '\t' << '\t' << 2 * csrmat->nnzs * 2 / (time_spmv_csr / nloops) / 1000 << " GFLOPS" << std::endl;
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if VEC_LENGTH == 256
    using Arch = xsimd::avx2;
#elif VEC_LENGTH == 512
    using Arch = xsimd::avx512f;
#endif

    std::vector<double> vtimethread(nthreads, 0.0);

    // SIMD warmup
    for (int i = 0; i < 100; i++)
        SPMV_SIMD<DATATYPE, INDEXTYPE, Arch>(vsimdmat, vstartrow, vtimethread, x, y2, nthreads);
    
    // SIMD test
    vtimethread.resize(nthreads, 0.0);
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nloops; i++)
        SPMV_SIMD<DATATYPE, INDEXTYPE, Arch>(vsimdmat, vstartrow, vtimethread, x, y2, nthreads);
    auto end2 = std::chrono::high_resolution_clock::now();
    double time_spmv_simd = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
    std::cout << "Time of SIMD SPMV:" << '\t' << time_spmv_simd / nloops << " us" << '\t' << '\t' << 2 * csrmat->nnzs * 2 / (time_spmv_simd / nloops) / 1000 << " GFLOPS" << std::endl;

#ifdef CSR_TIME_TEST
    std::cout << "Speedup:" << '\t' << '\t' << time_spmv_csr / time_spmv_simd << "x" << std::endl;
#endif

#ifdef THREAD_TIME_TEST
    std::cout << '\n';
    for (int i = 0; i < nthreads; i++)
        std::cout << "Thread " << i << ":" << '\t' 
                  << "time = " << vtimethread[i] / nloops << " us" << '\t' 
                  << "gflops = " << 2 * vsimdmat[i].nnzs * 2 / (vtimethread[i] / nloops) / 1000 << " GFLOPS" << '\t' 
                  << "nrows = " << vsimdmat[i].nrows << '\t' 
                  << "nnzs = " << vsimdmat[i].nnzs << '\t' 
                  << "cost = " << vsimdmat[i].cost 
                  << std::endl;
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // verify results
    SPMV_CSR_complex<DATATYPE>(csrmat, x, y1);
    SPMV_SIMD<DATATYPE, INDEXTYPE, Arch>(vsimdmat, vstartrow, vtimethread, x, y2);

    // compare y1 and y2
    for (int i = 0; i < csrmat->nrows * 2; i++) {
        if ( fabs(y1[i] - y2[i]) > precision && fabs(y1[i]-y2[i]) / fabs(y1[i]) > precision ) {
            std::cout << std::setprecision(20) << "Wrong result at row " << i << ": " << y1[i] << " vs " << y2[i] << std::endl;
        }
    }

    free(x);
    free(y1);
    free(y2);
    return 0;
}