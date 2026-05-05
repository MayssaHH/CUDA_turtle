#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM_Y 4
#define TILE_DIM_X 64

#define THRESHOLD 800000

// ─────────────────────────────────────────────────────────────
// WIDE KERNEL — original gpu2 kernel:
// ─────────────────────────────────────────────────────────────
__global__ void sptrsv_gpu3_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,
        unsigned int  levelSize,
        unsigned int  numCols)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= numCols || r >= levelSize) return;

    unsigned int i         = levelRows[r];
    unsigned int nB        = numCols;
    unsigned int row_start = L_r->rowPtrs[i];
    unsigned int row_end   = L_r->rowPtrs[i + 1];
    unsigned int row_len   = row_end - row_start;

    __shared__ unsigned int s_col[TILE_DIM_Y][TILE_DIM_X];
    __shared__ float        s_val[TILE_DIM_Y][TILE_DIM_X];
    __shared__ float        s_inv_diag[TILE_DIM_Y];

    // ── Pre-pass: find diagonal in one strided scan, no shared memory needed.
    // All 64 threads in x cooperate: each checks every TILE_DIM_X-th entry.
    // Exactly one thread will find col == i and write s_inv_diag.
    // All others skip the write — no contention, no atomics needed.
    if (threadIdx.x == 0) {
        s_inv_diag[threadIdx.y] = 1.0f;   // safe default
    }
    __syncthreads();

    for (unsigned int k = threadIdx.x; k < row_len; k += TILE_DIM_X) {
        if (L_r->colIdxs[row_start + k] == i) {
            float v = L_r->values[row_start + k];
            s_inv_diag[threadIdx.y] = 1.0f / (v != 0.0f ? v : 1.0f);
            // no break — only one entry matches, loop will finish cheaply
        }
    }

    // All threads must see s_inv_diag before the tile loop reads it
    __syncthreads();

    // ── Tile loop: pure accumulation, no diagonal check at all ──
    float sum = B->values[i * nB + b];

    for (unsigned int base = 0; base < row_len; base += TILE_DIM_X) {

        unsigned int k = base + threadIdx.x;
        if (k < row_len) {
            s_col[threadIdx.y][threadIdx.x] = L_r->colIdxs[row_start + k];
            s_val[threadIdx.y][threadIdx.x] = L_r->values[row_start + k];
        } else {
            s_col[threadIdx.y][threadIdx.x] = i;
            s_val[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        unsigned int tile_limit = min(TILE_DIM_X, row_len - base);
        for (unsigned int j = 0; j < tile_limit; ++j) {
            if (s_col[threadIdx.y][j] < i) {
                sum -= s_val[threadIdx.y][j] * X->values[s_col[threadIdx.y][j] * nB + b];
            }
        }

        __syncthreads();
    }

    X->values[i * nB + b] = sum * s_inv_diag[threadIdx.y];
}

// ─────────────────────────────────────────────────────────────
// THIN KERNEL — one thread per RHS column.
// ─────────────────────────────────────────────────────────────
__global__ void sptrsv_thin_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int  numCols)
{
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

// ─────────────────────────────────────────────────────────────
// HOST WRAPPER
// ─────────────────────────────────────────────────────────────
void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
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

    if (fabs(numLevels - numCols) >= THRESHOLD) {
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
            sptrsv_gpu3_kernel<<<gridDim, blockDim, 0, stream>>>(
                L_r,
                B,
                X,
                levelRows_d + levelStart,
                levelSize,
                numCols
            );
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
        
        // ── Level-Set method ──────────────────
    } else {
        const unsigned int blockSize = 256;
        const unsigned int gridSize  = (numCols + blockSize - 1) / blockSize;

        // Launch kernel
        sptrsv_thin_kernel<<<gridSize, blockSize>>>(L_r, B, X, numCols);

    }
    
    CUDA_ERROR_CHECK(cudaFree(levelRows_d));

    free(level);
    free(levelCount);
    free(levelOffsets);
    free(levelRows);
    free(fillPos);
}
