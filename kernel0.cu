// kernel0.cu

#include "common.h"

// This is the second GPU implementation of the sparse triangular solve (SpTRSV).
// In this version, we go beyond the first GPU idea where we only parallelized
// across the RHS columns, and we now try to recover some row parallelism too.
//
// Key idea:
// - In forward substitution, row i depends on some earlier rows j < i.
// - So in general we cannot solve all rows at the same time.
// - But also, not every row depends on every previous row.
// - Many rows are actually independent from each other and can be solved
//   together safely.
// - To exploit that, we compute a "level" for each row.
// - Rows in the same level do not depend on each other, so they can run in
//   parallel.
// - Then we solve the matrix level by level.
//
// What this file is doing overall:
// 1. On the host, analyze the sparse lower triangular matrix and compute the
//    dependency level of every row.
// 2. Group rows by level.
// 3. Launch one kernel per level.
// 4. Inside each kernel launch, parallelize in 2 dimensions:
//    - x-dimension: RHS columns
//    - y-dimension: rows inside the current level
//
// So compared to version 1:
// - v1 only had column parallelism
// - this version has column parallelism + row parallelism inside each level
// - but we still need a global synchronization between levels



__global__ void sptrsv_gpu0_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,
        unsigned int  levelSize,
        unsigned int  numCols)
{
    // Thread mapping:
    // - x dimension handles RHS columns
    // - y dimension handles rows inside the current level
    unsigned int b        = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int levelIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard against out-of-bounds threads
    if (b >= numCols || levelIdx >= levelSize) return;

    // Actual row in the matrix corresponding to this thread's level position
    unsigned int i  = levelRows[levelIdx];
    unsigned int nB = numCols;

    // Start from the RHS value B(i,b)
    float sum  = B->values[i * nB + b];

    // This will hold the diagonal entry L(i,i)
    float diag = 1.0f;

    // Traverse row i in CSR format
    for (unsigned int idx = L_r->rowPtrs[i]; idx < L_r->rowPtrs[i + 1]; ++idx) {
        unsigned int col = L_r->colIdxs[idx];
        float        val = L_r->values[idx];

        // If col < i, this contribution comes from a row that should already
        // have been solved in an earlier level, so we subtract it from the sum
        if (col < i) {
            sum -= val * X->values[col * nB + b];

        // If col == i, this is the diagonal element needed for the final divide
        } else if (col == i) {
            diag = (val != 0.0f) ? val : 1.0f;
        }
    }

    // Final forward substitution update for X(i,b)
    X->values[i * nB + b] = sum / diag;
}


void sptrsv_gpu0(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    // These two inputs are not needed in this version
    // The dependency analysis and the solve both rely on the CSR structure
    (void)L_c;
    (void)L_c_host;

    unsigned int n = L_r_host->numRows;

    // Step 1: compute the level of each row
    // level[i] = length of the longest dependency chain ending at row i
    // Since the matrix is lower triangular, dependencies col < i are already
    // known when we process row i from top to bottom
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i]; idx < L_r_host->rowPtrs[i + 1]; ++idx) {
            unsigned int col = L_r_host->colIdxs[idx];
            if (col < i) {
                // If row i depends on row col, then row i must be at least one
                // level after row col
                unsigned int candidate = level[col] + 1;
                if (candidate > level[i]) level[i] = candidate;
            }
        }
    }

    // Step 2: determine how many total levels we have
    unsigned int numLevels = 0;
    for (unsigned int i = 0; i < n; ++i)
        if (level[i] > numLevels) numLevels = level[i];
    numLevels++;

    // Step 3: count how many rows fall in each level
    unsigned int* levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i)
        levelCount[level[i]]++;

    // Step 4: prefix sum over level counts so we know where each level starts
    // inside the compact levelRows array
    unsigned int* levelOffsets = (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
    levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k)
        levelOffsets[k + 1] = levelOffsets[k] + levelCount[k];

    // Step 5: build levelRows
    // This stores the actual row indices, grouped level by level
    unsigned int* levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* fillPos   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int k = level[i];
        levelRows[levelOffsets[k] + fillPos[k]] = i;
        fillPos[k]++;
    }

    unsigned int* levelRows_d;
    CUDA_ERROR_CHECK(cudaMalloc((void**)&levelRows_d, n * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    // 2D block:
    // - x dimension for RHS columns
    // - y dimension for rows in the current level
    const dim3 blockDim(16, 16);

    // Step 6: solve level by level
    // We launch one kernel per level, and we synchronize after each launch
    // because the next level depends on the previous one being fully done
    for (unsigned int k = 0; k < numLevels; ++k) {
        unsigned int levelSize  = levelCount[k];
        unsigned int levelStart = levelOffsets[k];

        // Grid covers all (row, column) pairs in this level
        dim3 gridDim(
            (numCols   + blockDim.x - 1) / blockDim.x,
            (levelSize + blockDim.y - 1) / blockDim.y
        );

        // Launch the current level
        sptrsv_gpu0_kernel<<<gridDim, blockDim>>>(
            L_r, B, X,
            levelRows_d + levelStart,
            levelSize, numCols
        );

        // This synchronization is necessary for correctness:
        // rows in level k+1 may depend on rows from level k
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }

    // Cleanup
    CUDA_ERROR_CHECK(cudaFree(levelRows_d));
    free(level);
    free(levelCount);
    free(levelOffsets);
    free(levelRows);
    free(fillPos);
}
