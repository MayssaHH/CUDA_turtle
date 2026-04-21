// kernel2.cu
//
// Second optimization, builds on kernel1 (column-major layout) by adding
// shared memory tiling for the sparse inner loop.
//
// Problem in kernel1: during the sparse inner loop, thread (tx, ty) reads
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
// O(nnz) - a up to BLOCK_X-fold reduction in CSR global memory traffic.
//
// Syncthreads correctness: different ty-rows can have different nnz values and
// thus different tile counts. Each tx=0 thread writes its row's nnz into
// shm_nnz[ty]; all threads read the block-wide maximum and loop for
// ceil(maxNnz / BLOCK_X) tiles uniformly. Rows that have exhausted their
// nonzeros load a sentinel (col = UINT_MAX) for the remaining steps so no
// spurious contributions are accumulated, and all threads always reach both
// __syncthreads() calls inside the loop.
//
// Everything else - column-major indexing, level-set scheduling, grid/block
// dimensions, preprocess/postprocess - is identical to kernel1.

#include "common.h"

#define BLOCK_X 16
#define BLOCK_Y 16


// Each thread owns one (row i, RHS column b) pair.
// B and X are expected in column-major layout (same as kernel1).
__global__ void sptrsv_gpu2_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,
        unsigned int  levelSize,
        unsigned int  numRows,
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

    // column-major read for B; out-of-bounds threads carry a dummy 0
    float sum  = valid ? B->values[b * numRows + i] : 0.0f;
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

    // - phase 2: tiled sparse-row traversal
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
                    // column-major: retains kernel1's cache benefit for X reads
                    sum -= val * X->values[b * numRows + col];
                } else if (col == i) {
                    diag = (val != 0.0f) ? val : 1.0f;
                }
            }
        }
        __syncthreads();  // tile fully consumed before overwriting next tile
    }

    // write result in column-major layout
    if (valid)
        X->values[b * numRows + i] = sum / diag;
}


// Transpose helpers (identical to kernel1)
static void transposeToColMajor(const float* src, float* dst,
                                 unsigned int numRows, unsigned int numCols)
{
    for (unsigned int i = 0; i < numRows; ++i)
        for (unsigned int j = 0; j < numCols; ++j)
            dst[j * numRows + i] = src[i * numCols + j];
}

static void transposeToRowMajor(const float* src, float* dst,
                                 unsigned int numRows, unsigned int numCols)
{
    for (unsigned int j = 0; j < numCols; ++j)
        for (unsigned int i = 0; i < numRows; ++i)
            dst[i * numCols + j] = src[j * numRows + i];
}


// Static state (same pattern as kernel1, prefixed s2_ to avoid name clashes)
static CSRMatrix*    s2_L_r          = NULL;
static DenseMatrix*  s2_B            = NULL;
static DenseMatrix*  s2_X            = NULL;
static float*        s2_B_dev_vals   = NULL;
static float*        s2_X_dev_vals   = NULL;
static unsigned int* s2_levelRows_d  = NULL;
static unsigned int* s2_levelCount   = NULL;
static unsigned int* s2_levelOffsets = NULL;
static unsigned int  s2_numLevels    = 0;
static unsigned int  s2_numRows      = 0;
static unsigned int  s2_numCols      = 0;


// sptrsv_gpu2_preprocess  (identical logic to kernel1's preprocess)
void sptrsv_gpu2_preprocess(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                             CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int numRows = L_r_host->numRows;
    unsigned int n       = numRows;
    size_t       sz      = (size_t)numRows * numCols * sizeof(float);

    s2_L_r     = L_r;
    s2_B       = B;
    s2_X       = X;
    s2_numRows = numRows;
    s2_numCols = numCols;

    // pull device-side float pointers out of the GPU struct headers
    DenseMatrix B_hdr, X_hdr;
    CUDA_ERROR_CHECK(cudaMemcpy(&B_hdr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(&X_hdr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    s2_B_dev_vals = B_hdr.values;
    s2_X_dev_vals = X_hdr.values;

    // transpose B from row-major to column-major on the device
    float* tmp  = (float*)malloc(sz);
    float* B_cm = (float*)malloc(sz);
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, s2_B_dev_vals, sz, cudaMemcpyDeviceToHost));
    transposeToColMajor(tmp, B_cm, numRows, numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(s2_B_dev_vals, B_cm, sz, cudaMemcpyHostToDevice));
    free(B_cm);
    free(tmp);

    // level-set analysis: level[i] = longest dependency chain ending at row i
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
    s2_numLevels = numLevels;

    s2_levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i)
        s2_levelCount[level[i]]++;

    s2_levelOffsets = (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
    s2_levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k)
        s2_levelOffsets[k + 1] = s2_levelOffsets[k] + s2_levelCount[k];

    unsigned int* levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* fillPos   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int k = level[i];
        levelRows[s2_levelOffsets[k] + fillPos[k]] = i;
        fillPos[k]++;
    }

    CUDA_ERROR_CHECK(cudaMalloc((void**)&s2_levelRows_d, n * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(s2_levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    free(level);
    free(levelRows);
    free(fillPos);
}


// sptrsv_gpu2_solve  - the timed region
void sptrsv_gpu2_solve()
{
    const dim3 blockDim(BLOCK_X, BLOCK_Y);

    for (unsigned int k = 0; k < s2_numLevels; ++k) {
        unsigned int levelSize  = s2_levelCount[k];
        unsigned int levelStart = s2_levelOffsets[k];

        dim3 gridDim(
            (s2_numCols + blockDim.x - 1) / blockDim.x,
            (levelSize  + blockDim.y - 1) / blockDim.y
        );

        sptrsv_gpu2_kernel<<<gridDim, blockDim>>>(
            s2_L_r, s2_B, s2_X,
            s2_levelRows_d + levelStart,
            levelSize, s2_numRows, s2_numCols
        );

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
}


// sptrsv_gpu2_postprocess  (identical logic to kernel1's postprocess)
void sptrsv_gpu2_postprocess()
{
    size_t sz  = (size_t)s2_numRows * s2_numCols * sizeof(float);
    float* tmp = (float*)malloc(sz);
    float* buf = (float*)malloc(sz);

    // transpose X back to row-major so verify() sees the expected layout
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, s2_X_dev_vals, sz, cudaMemcpyDeviceToHost));
    transposeToRowMajor(tmp, buf, s2_numRows, s2_numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(s2_X_dev_vals, buf, sz, cudaMemcpyHostToDevice));

    // restore B to row-major so the caller's buffer is left unchanged
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, s2_B_dev_vals, sz, cudaMemcpyDeviceToHost));
    transposeToRowMajor(tmp, buf, s2_numRows, s2_numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(s2_B_dev_vals, buf, sz, cudaMemcpyHostToDevice));

    free(tmp);
    free(buf);

    CUDA_ERROR_CHECK(cudaFree(s2_levelRows_d));
    free(s2_levelCount);
    free(s2_levelOffsets);

    s2_L_r = NULL; s2_B = NULL; s2_X = NULL;
    s2_B_dev_vals = NULL; s2_X_dev_vals = NULL;
    s2_levelRows_d = NULL; s2_levelCount = NULL; s2_levelOffsets = NULL;
    s2_numLevels = 0; s2_numRows = 0; s2_numCols = 0;
}
