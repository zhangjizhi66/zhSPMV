
#ifndef CSR_H
#define CSR_H

#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <charconv>
#include <set>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace CSR {

    template <typename T>
    class CSRMAT {
    public:
        CSRMAT(std::string MtxFileName);
        CSRMAT(int rows, int cols, int nonzeros);
        CSRMAT(const CSRMAT &) = delete;  // 禁止拷贝构造函数
        CSRMAT &operator=(const CSRMAT &) = delete;  // 禁止
        ~CSRMAT() {
            delete[] nnz_val;
            delete[] col_idx;
            delete[] row_ptr;
        }

        int nrows;
        int ncols;
        int nnzs;
        
        T *nnz_val;
        int *col_idx;
        int *row_ptr;
    };

    template <typename T>
    CSRMAT<T>::CSRMAT(std::string MtxFileName) {
        std::ifstream mtxfile(MtxFileName);
        if (!mtxfile.is_open()) {
            std::cerr << "Error opening file: " << MtxFileName << std::endl;
            exit(1);
        }

        std::string line;
        while (std::getline(mtxfile, line)) {
            if (line[0] == '%') continue;
            std::istringstream iss(line);
            iss >> nrows >> ncols >> nnzs;
            break;
        }

        if ( nrows != ncols ) {
            std::cerr << "Error: Matrix is not square." << std::endl;
            exit(1);
        }

        nnz_val = new T[nnzs];
        col_idx = new int[nnzs];
        row_ptr = new int[nrows + 1];
        row_ptr[0] = 0;

        struct COOPNT {
            int row;
            int col;
            T val;
        };
        std::vector<COOPNT> coo_list;

        while (std::getline(mtxfile, line)) {
            std::istringstream iss(line);
            int row, col;
            T val;
            iss >> row >> col >> val;
            row--; // Convert to 0-based index
            col--; // Convert to 0-based index
            coo_list.push_back({row, col, val});
        }

        std::sort(coo_list.begin(), coo_list.end(), [](const COOPNT &a, const COOPNT &b) {
            if ( a.row == b.row && a.col == b.col ) {
                std::cerr << "Error: Duplicate entry found in COO format." << std::endl;
                exit(1);
            }
            return a.row < b.row || (a.row == b.row && a.col < b.col);
        });

        int current_row = 0;
        for (int i = 0; i < nnzs; i++) {
            while (current_row < coo_list[i].row)
                row_ptr[++current_row] = i;

            nnz_val[i] = coo_list[i].val;
            col_idx[i] = coo_list[i].col;
        }
        while (current_row < nrows)
            row_ptr[++current_row] = nnzs;
        row_ptr[nrows] = nnzs;

        mtxfile.close();
    }

    template <typename T>
    CSRMAT<T>::CSRMAT(int rows, int cols, int nonzeros) : nrows(rows), ncols(cols), nnzs(nonzeros) {
        if ( nrows <= 0 || ncols <= 0 || nnzs <= 0 || nnzs > int64_t(nrows) * ncols ) {
            std::cerr << "Invalid matrix size!" << std::endl;
            exit(1);
        }
        
        nnz_val = new T[nnzs];
        col_idx = new int[nnzs];
        row_ptr = new int[nrows + 1];
        
        // 使用集合确保唯一性
        std::set<std::pair<int, int>> positions;
        
        // 随机选择行分布
        std::vector<int> row_distribution(nrows, 0);
        
        // 先确保每行至少有一个非零元
        if (nonzeros >= nrows) {
            std::mt19937 gen(0);
            std::uniform_int_distribution<int> col_dist(0, cols-1);
            
            for (int i = 0; i < nrows; i++) {
                int col;
                do {
                    col = col_dist(gen);
                } while (positions.find({i, col}) != positions.end());
                
                positions.insert({i, col});
                row_distribution[i]++;
            }
        }
        
        // 生成剩余的非零元
        std::mt19937 gen(0);
        std::uniform_int_distribution<int> row_dist(0, nrows-1);
        std::uniform_int_distribution<int> col_dist(0, cols-1);
        
        for (int i = positions.size(); i < nonzeros; i++) {
            int row, col;
            do {
                row = row_dist(gen);
                col = col_dist(gen);
            } while (positions.find({row, col}) != positions.end());
            
            positions.insert({row, col});
            row_distribution[row]++;
        }
        
        // 按行排序
        std::vector<std::pair<int, int>> coo_list(positions.begin(), positions.end());
        std::sort(coo_list.begin(), coo_list.end(), 
            [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            });
        
        // 构建CSR格式
        row_ptr[0] = 0;
        int current_row = 0;
        
        for (int i = 0; i < nnzs; i++) {
            while (current_row < coo_list[i].first)
                row_ptr[++current_row] = i;
                
            nnz_val[i] = static_cast<T>(0.1);  // 固定非零元值为0.1
            col_idx[i] = coo_list[i].second;
        }
        
        // 填充剩余的行指针
        while (current_row < nrows)
            row_ptr[++current_row] = nnzs;
            
        row_ptr[nrows] = nnzs;
    }

    template <typename T>
    void ParseArgs(int argc, char *argv[], CSRMAT<T>* &csrmat, int &nloops, int &nthreads, double &precision) {
        if ( argc < 2 ) {
            std::cerr << "Usage 1: " << argv[0] << " <matrix_file> [nloops] [nthreads] [precision]" << std::endl;
            std::cerr << "Example 1: " << argv[0] << " matrix.mtx 1000 4 1e-14" << std::endl;
            std::cerr << "Usage 2: " << argv[0] << " <nrows> <ncols> <nnzs> [nloops] [nthreads] [precision]" << std::endl;
            std::cerr << "Example 2: " << argv[0] << " 10000 10000 100000 1000 4 1e-14" << std::endl;
            exit(1);
        }

        int nrows;
        auto [ptr, ec] = std::from_chars(argv[1], argv[1] + strlen(argv[1]), nrows);

        if ( ec == std::errc::invalid_argument ) {
            std::string MtxFileName = argv[1];
            csrmat = new CSRMAT<T>(MtxFileName);
            if ( argc > 2 ) nloops = std::stoi(argv[2]);
            if ( argc > 3 ) nthreads = std::stoi(argv[3]);
            if ( argc > 4 ) precision = std::stod(argv[4]);
        }
        else {
            if ( argc < 4 ) {
                std::cerr << "Usage: " << argv[0] << " <nrows> <ncols> <nnzs> [nloops] [nthreads] [precision]" << std::endl;
                std::cerr << "Example: " << argv[0] << " 10000 10000 100000 1000 4 1e-14" << std::endl;
                exit(1);
            }
            csrmat = new CSRMAT<T>(nrows, std::stoi(argv[2]), std::stoi(argv[3]));
            if ( argc > 4 ) nloops = std::stoi(argv[4]);
            if ( argc > 5 ) nthreads = std::stoi(argv[5]);
            if ( argc > 6 ) precision = std::stod(argv[6]);
        }
    }

    template <typename T>
    void SPMV_CSR(CSRMAT<T> *csrmat, T *x, T *y, int nthreads = 1) {
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
        for (int i = 0; i < csrmat->nrows; i++) {
            y[i] = 0;
            for (int j = csrmat->row_ptr[i]; j < csrmat->row_ptr[i + 1]; j++)
                y[i] += csrmat->nnz_val[j] * x[csrmat->col_idx[j]];
        }
    }

}

#endif // SPMV_CSR_H