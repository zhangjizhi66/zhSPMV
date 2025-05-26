
#ifndef SPMV_CSR_H
#define SPMV_CSR_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

template <typename T>
class CSRMAT {
public:
    CSRMAT(std::string MtxFileName);
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
        if (line[0] == '%') continue; // Skip comments
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
void SPMV_CSR(CSRMAT<T> &csrmat, T *x, T *y, int nthreads = 1) {
    //#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    for (int i = 0; i < csrmat.nrows; i++) {
        y[i] = 0;
        for (int j = csrmat.row_ptr[i]; j < csrmat.row_ptr[i + 1]; j++)
            y[i] += csrmat.nnz_val[j] * x[csrmat.col_idx[j]];
    }
}

#endif // SPMV_CSR_H