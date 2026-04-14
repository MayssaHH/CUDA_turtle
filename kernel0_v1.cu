// kernel0_v1.cu

#include "common.h"

// This is the first GPU implementation of the sparse triangular solve (SpTRSV).
// The goal is to parallelize the CPU forward substitution algorithm while
// preserving correctness.
//
// Key idea:
// - The system being solved is: L * X = B, where L is lower triangular.
// - Forward substitution introduces a dependency along rows: X(i,b) depends on
//   X(0..i-1, b), which prevents parallelization across rows.
// - However, each RHS column b is independent, so we parallelize across columns.
//
// Strategy in this version:
// - Assign one thread per RHS column.
// - Each thread performs the full forward substitution for its column.
// - No synchronization is needed, since threads do not share data dependencies.

__global__ void sptrsv_gpu0_kernel_v1(CSRMatrix* L_r, DenseMatrix* B,
                                      DenseMatrix* X, unsigned int numCols) {

    // Each thread is responsible for solving one RHS column b
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;

    // Number of rows in L (size of system)
    unsigned int n  = L_r->numRows;

    // Number of RHS columns
    unsigned int nB = numCols;

    // Guard: ensure thread corresponds to a valid column
    if (b >= nB) return;

    // Forward substitution over rows (must remain sequential)
    // Row i depends only on previously computed rows 0..i-1
    for (unsigned int i = 0; i < n; ++i) {

        // Initialize accumulator with RHS value B(i, b)
        // This will be reduced by subtracting known contributions
        float sum = B->values[i * nB + b];

        // Variable to store the diagonal entry L(i,i)
        // This is required at the end to solve for X(i,b)
        float diag = 1.0f;

        // Traverse all nonzero entries in row i of L (CSR format)
        for (unsigned int idx = L_r->rowPtrs[i];
             idx < L_r->rowPtrs[i + 1]; ++idx) {

            // Column index and value of current nonzero
            unsigned int col = L_r->colIdxs[idx];
            float val = L_r->values[idx];

            // If col < i:
            // This corresponds to a previously solved variable X(col, b),
            // so we subtract its contribution from the sum
            if (col < i) {
                sum -= val * X->values[col * nB + b];

            // If col == i:
            // This is the diagonal element L(i,i)
            // We store it for the final division
            } else if (col == i) {
                diag = (val != 0.0f) ? val : 1.0f;
            }
        }

        // After removing all lower-triangular contributions:
        // sum = B(i,b) - Σ L(i,j)*X(j,b), j < i
        // So we solve:
        // X(i,b) = sum / L(i,i)
        X->values[i * nB + b] = sum / diag;
    }
}

// Host wrapper for launching the GPU kernel
void sptrsv_gpu0_v1(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B,
                    DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host,
                    unsigned int numCols) {

    // These inputs are not used in this version
    // CSR format is sufficient for row-wise traversal
    (void)L_c;
    (void)L_c_host;
    (void)L_r_host;

    // Configure kernel launch:
    // - Threads correspond to RHS columns
    // - Use a standard 1D grid
    const unsigned int blockSize = 256;
    const unsigned int gridSize  = (numCols + blockSize - 1) / blockSize;

    // Launch kernel
    sptrsv_gpu0_kernel_v1<<<gridSize, blockSize>>>(L_r, B, X, numCols);

    // Check for runtime errors
    CUDA_ERROR_CHECK(cudaGetLastError());

    // Ensure kernel completion before returning
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
