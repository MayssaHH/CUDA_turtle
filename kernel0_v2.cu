// kernel0_v2.cu

#include "common.h"

// This is the second GPU implementation of the sparse triangular solve (SpTRSV).
// In this version, we go beyond simple column-wise parallelism and exploit
// additional parallelism across rows using a level-set strategy.
//
// Key idea:
// - In forward substitution, row i depends on rows j < i such that L(i,j) ≠ 0.
// - These dependencies form a directed acyclic graph (DAG).
// - We assign each row a "level" such that rows in the same level have no
//   dependencies on each other and can therefore be solved in parallel.
// - The solve proceeds level by level: within each level, all rows are processed
//   in parallel, and synchronization is enforced between levels.
//
// Parallelization strategy:
// - Parallelize across RHS columns (b) as before.
// - Additionally parallelize across rows within the same level.
// - Each thread computes one (row, column) pair.

__global__ void sptrsv_gpu0_kernel_v2(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,
        unsigned int  levelSize,
        unsigned int  numCols)
{
    // Thread mapping:
    // - x-dimension handles RHS columns
    // - y-dimension handles rows within the current level
    unsigned int b        = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int levelIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard against out-of-bounds threads
    if (b >= numCols || levelIdx >= levelSize) return;

    // Actual row index in the matrix
    unsigned int i  = levelRows[levelIdx];
    unsigned int nB = numCols;

    // Initialize with RHS value B(i, b)
    float sum  = B->values[i * nB + b];

    // Diagonal entry L(i,i)
    float diag = 1.0f;

    // Traverse nonzero elements of row i
    for (unsigned int idx = L_r->rowPtrs[i];
                      idx < L_r->rowPtrs[i + 1]; ++idx) {

        unsigned int col = L_r->colIdxs[idx];
        float        val = L_r->values[idx];

        // Contributions from previously solved rows (earlier levels)
        if (col < i) {
            sum -= val * X->values[col * nB + b];

        // Diagonal entry
        } else if (col == i) {
            diag = (val != 0.0f) ? val : 1.0f;
        }
    }

    // Solve for X(i,b)
    X->values[i * nB + b] = sum / diag;
}


// Host wrapper
void sptrsv_gpu0_v2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                    CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    // Number of rows in the system
    unsigned int n = L_r_host->numRows;

    // Step 1: Compute level of each row (dependency analysis)
    // level[i] = depth of row i in dependency graph
    // (the maximum number of sequential dependencies before i)
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));

    // Forward pass: since matrix is lower triangular,
    // dependencies col < i are already processed
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i];
                          idx < L_r_host->rowPtrs[i + 1]; ++idx) {

            unsigned int col = L_r_host->colIdxs[idx];

            if (col < i) {
                unsigned int candidate = level[col] + 1;
                if (candidate > level[i]) level[i] = candidate;
            }
        }
    }

    // Determine total number of levels
    unsigned int numLevels = 0;
    for (unsigned int i = 0; i < n; ++i) {
        if (level[i] > numLevels) numLevels = level[i];
    }
    numLevels++; // levels are 0-indexed

    // Step 2: Count how many rows belong to each level
    unsigned int* levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        levelCount[level[i]]++;
    }

    // Step 3: Compute offsets for each level (prefix sum)
    unsigned int* levelOffsets =
        (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));

    levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k) {
        levelOffsets[k + 1] = levelOffsets[k] + levelCount[k];
    }

    // Step 4: Build levelRows array (rows grouped by level)
    unsigned int* levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* fillPos   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));

    for (unsigned int i = 0; i < n; ++i) {
        unsigned int k = level[i];
        levelRows[levelOffsets[k] + fillPos[k]] = i;
        fillPos[k]++;
    }

    // Step 5: Copy levelRows to GPU
    unsigned int* levelRows_d;
    CUDA_ERROR_CHECK(cudaMalloc((void**)&levelRows_d, n * sizeof(unsigned int)));

    CUDA_ERROR_CHECK(cudaMemcpy(levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    // 2D thread block:
    // - x: RHS columns
    // - y: rows within level
    const dim3 blockDim(16, 16);

    // Step 6: Process each level sequentially
    for (unsigned int k = 0; k < numLevels; ++k) {

        unsigned int levelSize  = levelCount[k];
        unsigned int levelStart = levelOffsets[k];

        // Grid dimensions cover all (row, column) pairs in this level
        dim3 gridDim(
            (numCols   + blockDim.x - 1) / blockDim.x,
            (levelSize + blockDim.y - 1) / blockDim.y
        );

        // Launch kernel for this level
        sptrsv_gpu0_kernel_v2<<<gridDim, blockDim>>>(
            L_r,
            B,
            X,
            levelRows_d + levelStart,
            levelSize,
            numCols
        );

        // Ensure all rows in this level are completed before next level
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
