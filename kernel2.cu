#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_DIM_Y 4
#define TILE_DIM_X 64

__global__ void sptrsv_gpu2_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,
        unsigned int  levelSize,
        unsigned int  numCols)
{
    // x-dimension: RHS columns
    // y-dimension: rows within the current level
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= numCols || r >= levelSize) return;

    unsigned int i  = levelRows[r];
    unsigned int nB = numCols;

    unsigned int row_start = L_r->rowPtrs[i];
    unsigned int row_end   = L_r->rowPtrs[i + 1];
    unsigned int row_len   = row_end - row_start;

    // Each row in the y-dimension has its own shared memory slice.
    // Layout: s_col[threadIdx.y][BLOCK_DIM], s_val[threadIdx.y][BLOCK_DIM]
    // This way each row's tile is independent and there's no cross-row corruption.
    __shared__ unsigned int s_col[TILE_DIM_Y][TILE_DIM_X];
    __shared__ float        s_val[TILE_DIM_Y][TILE_DIM_X];

    float sum  = B->values[i * nB + b];
    float diag = 1.0f;

    for (unsigned int base = 0; base < row_len; base += TILE_DIM_X) {

        // Step 1: Cooperative load — all threads in x-dimension load one tile
        // entry for their own row (threadIdx.y). No cross-row dependency.
        unsigned int k = base + threadIdx.x;
        if (k < row_len) {
            s_col[threadIdx.y][threadIdx.x] = L_r->colIdxs[row_start + k];
            s_val[threadIdx.y][threadIdx.x] = L_r->values[row_start + k];
        } else {
            // Pad with a sentinel so the compute loop can run without branching
            // on tile_limit (optional but clean)
            s_col[threadIdx.y][threadIdx.x] = i; // diagonal sentinel — won't affect sum
            s_val[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Step 2: Process the tile for this row
        unsigned int tile_limit = min(TILE_DIM_X, row_len - base);

        for (unsigned int j = 0; j < tile_limit; ++j) {
            unsigned int col = s_col[threadIdx.y][j];
            float        val = s_val[threadIdx.y][j];

            if (col < i) {
                sum -= val * X->values[col * nB + b];
            } else if (col == i) {
                diag = val;
                if (diag == 0.0f) {
                    diag = 1.0f;
                }
            }
        }

        __syncthreads();
    }

    X->values[i * nB + b] = sum / diag;
}

// Host wrapper
void sptrsv_gpu2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
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
    const dim3 blockDim(64, 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreate(&event);

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
        sptrsv_gpu2_kernel<<<gridDim, blockDim, 0, stream>>>(
            L_r,
            B,
            X,
            levelRows_d + levelStart,
            levelSize,
            numCols
        );

        // Ensure all rows in this level are completed before next level
        CUDA_ERROR_CHECK(cudaGetLastError());
    }

    // Single sync at the very end
    cudaStreamSynchronize(stream);

    // Cleanup
    CUDA_ERROR_CHECK(cudaFree(levelRows_d));

    free(level);
    free(levelCount);
    free(levelOffsets);
    free(levelRows);
    free(fillPos);
}
