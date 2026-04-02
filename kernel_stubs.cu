// Temporary stubs when kernel0.cu–kernel3.cu are not in the tree.
// CPU path (-s) works; do not use -0..-3 with this build.

#include "common.h"

void sptrsv_gpu0(CSCMatrix*, CSRMatrix*, DenseMatrix*, DenseMatrix*, CSCMatrix*, CSRMatrix*, unsigned int) {}
void sptrsv_gpu1(CSCMatrix*, CSRMatrix*, DenseMatrix*, DenseMatrix*, CSCMatrix*, CSRMatrix*, unsigned int) {}
void sptrsv_gpu2(CSCMatrix*, CSRMatrix*, DenseMatrix*, DenseMatrix*, CSCMatrix*, CSRMatrix*, unsigned int) {}
void sptrsv_gpu3(CSCMatrix*, CSRMatrix*, DenseMatrix*, DenseMatrix*, CSCMatrix*, CSRMatrix*, unsigned int) {}
