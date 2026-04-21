// kernel3.cu
//
// Third optimization: shared memory tiling for the sparse inner loop,
// built on top of kernel0_v2 (level-set scheduling, row-major layout).
//
// This is the row-major counterpart of kernel2. The column-major layout
// change from kernel1/kernel2 is NOT applied here; B and X remain in
// row-major order throughout (no preprocess/postprocess transpose).
//
// Problem in kernel0_v2: during the sparse inner loop, thread (tx, ty) reads
// L_r->colIdxs[idx] and L_r->values[idx] from global memory for each nonzero
// in row i. All 16 tx-threads sharing the same ty (same row i) iterate over
// the exact same nonzeros - the same global addresses are read 16 times, once
// per tx-thread. This is BLOCK_X-fold redundant global memory traffic for CSR.
//
// Fix: tile the sparse row using shared memory. For each tile of BLOCK_X
// consecutive nonzeros in row i:
//   1. The BLOCK_X tx-threads sharing ty cooperate: each thread tx loads one
//      (colIdx, value) entry at offset (t*BLOCK_X + tx) from the row start
//      into shm_col[ty][tx] and shm_val[ty][tx].
//   2. __syncthreads() - tile is fully in shared memory.
//   3. All BLOCK_X tx-threads for this ty-row reuse shm_col[ty][0..tileLen-1]
//      and shm_val[ty][0..tileLen-1] while computing contributions to sum.
//   4. __syncthreads() - tile fully consumed before overwriting next tile.
//
// This reduces global reads of colIdxs/values from O(nnz * BLOCK_X) to
// O(nnz) - up to a BLOCK_X-fold reduction in CSR global memory traffic.
//
// Syncthreads correctness: different ty-rows can have different nnz values and
// thus different tile counts. Each tx=0 thread writes its row's nnz into
// shm_nnz[ty]; all threads read the block-wide maximum and loop for
// ceil(maxNnz / BLOCK_X) tiles uniformly. Rows that have exhausted their
// nonzeros load a sentinel (col = UINT_MAX) for the remaining steps so no
// spurious contributions are accumulated, and all threads always reach both
// __syncthreads() calls inside the loop.
//
// Level-set scheduling, grid/block dimensions, and row-major index arithmetic
// are identical to kernel0_v2.

#include "common.h"

#define BLOCK_X 16
#define BLOCK_Y 16


// Each thread owns one (row i, RHS column b) pair.
// B and X are in row-major layout: element (row, col) -> row*numCols + col.
__global__ void sptrsv_gpu3_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,
        unsigned int  levelSize,
        unsigned int  numCols)
{
    // per-ty-row nnz used to compute the block-wide tile count
    __shared__ unsigned int shm_nnz[BLOCK_Y];
    // tile buffers: shm_col[ty][tx] and shm_val[ty][tx] hold one tile of
    // BLOCK_X nonzeros for each ty-row simultaneously
    __shared__ unsigned int shm_col[BLOCK_Y][BLOCK_X];
    __shared__ float        shm_val[BLOCK_Y][BLOCK_X];

    unsigned int tx       = threadIdx.x;
    unsigned int ty       = threadIdx.y;
    unsigned int b        = blockIdx.x * BLOCK_X + tx;
    unsigned int levelIdx = blockIdx.y * BLOCK_Y + ty;

    // separate flags so out-of-bounds threads still participate in every
    // __syncthreads() without touching undefined memory
    bool ty_valid = (levelIdx < levelSize);
    bool tx_valid = (b < numCols);
    bool valid    = ty_valid && tx_valid;

    unsigned int i        = ty_valid ? levelRows[levelIdx] : 0u;
    unsigned int rowStart = ty_valid ? L_r->rowPtrs[i]     : 0u;
    unsigned int rowEnd   = ty_valid ? L_r->rowPtrs[i + 1] : 0u;
    unsigned int nnz      = rowEnd - rowStart;

    // row-major read for B; out-of-bounds threads carry a dummy 0
    float sum  = valid ? B->values[i * numCols + b] : 0.0f;
    float diag = 1.0f;

    // phase 1: share nnz to agree on tile count
    // only tx=0 writes; all threads with the same ty have identical nnz
    if (tx == 0)
        shm_nnz[ty] = nnz;
    __syncthreads();

    // every thread computes maxNnz independently (sequential scan, no atomic)
    unsigned int maxNnz = 0;
    for (unsigned int r = 0; r < BLOCK_Y; ++r)
        if (shm_nnz[r] > maxNnz) maxNnz = shm_nnz[r];

    unsigned int numTiles = (maxNnz + BLOCK_X - 1) / BLOCK_X;

    // phase 2: tiled sparse-row traversal
    for (unsigned int t = 0; t < numTiles; ++t) {

        unsigned int localIdx = t * BLOCK_X + tx;

        // cooperative load: thread tx loads the (t*BLOCK_X + tx)-th nonzero
        // of its ty-row; sentinel for rows that have run out of nonzeros
        if (ty_valid && localIdx < nnz) {
            shm_col[ty][tx] = L_r->colIdxs[rowStart + localIdx];
            shm_val[ty][tx] = L_r->values[rowStart + localIdx];
        } else {
            // UINT_MAX is always > i, so the compute phase skips it cleanly
            shm_col[ty][tx] = 0xFFFFFFFFu;
            shm_val[ty][tx] = 0.0f;
        }
        __syncthreads();  // tile fully loaded before any thread reads it

        // compute: all valid threads reuse the shared tile for their ty-row;
        // rows past their own tile count get tileLen = 0 and skip the loop
        if (valid) {
            unsigned int tileOffset = t * BLOCK_X;
            unsigned int remaining  = (nnz > tileOffset) ? (nnz - tileOffset) : 0u;
            unsigned int tileLen    = (remaining < (unsigned int)BLOCK_X)
                                      ? remaining : (unsigned int)BLOCK_X;

            for (unsigned int s = 0; s < tileLen; ++s) {
                unsigned int col = shm_col[ty][s];
                float        val = shm_val[ty][s];

                if (col < i) {
                    // row-major read: col*numCols + b (same as kernel0_v2)
                    sum -= val * X->values[col * numCols + b];
                } else if (col == i) {
                    diag = (val != 0.0f) ? val : 1.0f;
                }
            }
        }
        __syncthreads();  // tile fully consumed before overwriting next tile
    }

    // write result in row-major layout
    if (valid)
        X->values[i * numCols + b] = sum / diag;
}


// Static state shared between preprocess, solve, and postprocess
static CSRMatrix*    s3_L_r          = NULL;
static DenseMatrix*  s3_B            = NULL;
static DenseMatrix*  s3_X            = NULL;
static unsigned int* s3_levelRows_d  = NULL;
static unsigned int* s3_levelCount   = NULL;
static unsigned int* s3_levelOffsets = NULL;
static unsigned int  s3_numLevels    = 0;
static unsigned int  s3_numCols      = 0;


// sptrsv_gpu3_preprocess
// Level-set analysis and levelRows upload only; no layout change needed.
void sptrsv_gpu3_preprocess(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                             CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    s3_L_r     = L_r;
    s3_B       = B;
    s3_X       = X;
    s3_numCols = numCols;

    // level[i] = longest dependency chain ending at row i
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));
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

    unsigned int numLevels = 0;
    for (unsigned int i = 0; i < n; ++i)
        if (level[i] > numLevels) numLevels = level[i];
    numLevels++;
    s3_numLevels = numLevels;

    s3_levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i)
        s3_levelCount[level[i]]++;

    s3_levelOffsets = (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
    s3_levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k)
        s3_levelOffsets[k + 1] = s3_levelOffsets[k] + s3_levelCount[k];

    unsigned int* levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* fillPos   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int k = level[i];
        levelRows[s3_levelOffsets[k] + fillPos[k]] = i;
        fillPos[k]++;
    }

    CUDA_ERROR_CHECK(cudaMalloc((void**)&s3_levelRows_d, n * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(s3_levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    free(level);
    free(levelRows);
    free(fillPos);
}


// sptrsv_gpu3_solve  - the timed region
void sptrsv_gpu3_solve()
{
    const dim3 blockDim(BLOCK_X, BLOCK_Y);

    for (unsigned int k = 0; k < s3_numLevels; ++k) {
        unsigned int levelSize  = s3_levelCount[k];
        unsigned int levelStart = s3_levelOffsets[k];

        dim3 gridDim(
            (s3_numCols + blockDim.x - 1) / blockDim.x,
            (levelSize  + blockDim.y - 1) / blockDim.y
        );

        sptrsv_gpu3_kernel<<<gridDim, blockDim>>>(
            s3_L_r, s3_B, s3_X,
            s3_levelRows_d + levelStart,
            levelSize, s3_numCols
        );

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
}


// sptrsv_gpu3_postprocess  - free level-set state; no transpose needed
void sptrsv_gpu3_postprocess()
{
    CUDA_ERROR_CHECK(cudaFree(s3_levelRows_d));
    free(s3_levelCount);
    free(s3_levelOffsets);

    s3_L_r = NULL; s3_B = NULL; s3_X = NULL;
    s3_levelRows_d = NULL; s3_levelCount = NULL; s3_levelOffsets = NULL;
    s3_numLevels = 0; s3_numCols = 0;
}
