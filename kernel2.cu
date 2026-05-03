#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM_X 64
#define TILE_DIM_Y 4

// Threshold below which levels get batched into a single launch.
// Wide levels (>= threshold) are launched individually as before.
#define MERGE_THRESHOLD 256

// ─────────────────────────────────────────────────────────────
// Kernel for a single WIDE level (unchanged from gpu2).
// ─────────────────────────────────────────────────────────────
__global__ void sptrsv_wide_kernel(
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

    unsigned int i        = levelRows[r];
    unsigned int nB       = numCols;
    unsigned int row_start = L_r->rowPtrs[i];
    unsigned int row_end   = L_r->rowPtrs[i + 1];
    unsigned int row_len   = row_end - row_start;

    __shared__ unsigned int s_col[TILE_DIM_Y][TILE_DIM_X];
    __shared__ float        s_val[TILE_DIM_Y][TILE_DIM_X];

    float sum  = B->values[i * nB + b];
    float diag = 1.0f;

    for (unsigned int base = 0; base < row_len; base += TILE_DIM_X) {
        unsigned int k = base + threadIdx.x;
        if (k < row_len) {
            s_col[threadIdx.y][threadIdx.x] = L_r->colIdxs[row_start + k];
            s_val[threadIdx.y][threadIdx.x] = L_r->values[row_start + k];
        } else {
            s_col[threadIdx.y][threadIdx.x] = i + 1;
            s_val[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        unsigned int tile_limit = min(TILE_DIM_X, row_len - base);
        for (unsigned int j = 0; j < tile_limit; ++j) {
            unsigned int col = s_col[threadIdx.y][j];
            float        val = s_val[threadIdx.y][j];
            if      (col < i)  sum -= val * X->values[col * nB + b];
            else if (col == i) diag = (val != 0.0f) ? val : 1.0f;
        }
        __syncthreads();
    }

    X->values[i * nB + b] = sum / diag;
}

// ─────────────────────────────────────────────────────────────
// Kernel for a BATCH of thin levels.
//
// Layout of batchRows[0..batchSize-1]:
//   group 0: rows of original level k0   (groupSizes[0] entries)
//   group 1: rows of original level k0+1 (groupSizes[1] entries)
//   ...
//   group G-1: last merged level
//
// The kernel uses a 1D grid over RHS columns only.
// Each thread block handles ONE column range and iterates over
// ALL groups sequentially, using __syncthreads() between groups
// to ensure group g is fully written to X before group g+1 reads it.
//
// This works because all threads in every block participate in
// every __syncthreads() call — the block never exits early.
// ─────────────────────────────────────────────────────────────
__global__ void sptrsv_thin_batch_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* batchRows,
        unsigned int* groupOffsets,
        unsigned int  numGroups,
        unsigned int  batchSize,
        unsigned int  numCols)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= numCols) return;

    unsigned int nB = numCols;

    // When all groups have size 1, each thread independently walks
    // all rows in dependency order. No cross-thread communication needed
    // because thread b only ever reads and writes column b of X.
    // The sequential row order guarantees correctness without any barriers.
    for (unsigned int idx = 0; idx < batchSize; ++idx) {
        unsigned int i         = batchRows[idx];
        unsigned int row_start = L_r->rowPtrs[i];
        unsigned int row_end   = L_r->rowPtrs[i + 1];

        float sum  = B->values[i * nB + b];
        float diag = 1.0f;

        for (unsigned int j = row_start; j < row_end; ++j) {
            unsigned int col = L_r->colIdxs[j];
            float        val = L_r->values[j];
            if      (col < i)  sum -= val * X->values[col * nB + b];
            else if (col == i) diag = (val != 0.0f) ? val : 1.0f;
        }

        X->values[i * nB + b] = sum / diag;
    }
}


void sptrsv_gpu2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    // ── Level analysis ────────────────────────────────────────────────────
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i];
                          idx < L_r_host->rowPtrs[i + 1]; ++idx) {
            unsigned int col = L_r_host->colIdxs[idx];
            if (col < i) {
                unsigned int c = level[col] + 1;
                if (c > level[i]) level[i] = c;
            }
        }
    }

    unsigned int numLevels = 0;
    for (unsigned int i = 0; i < n; ++i)
        if (level[i] > numLevels) numLevels = level[i];
    numLevels++;

    unsigned int* levelCount   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    unsigned int* levelOffsets = (unsigned int*)malloc((numLevels+1) * sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) levelCount[level[i]]++;
    levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k)
        levelOffsets[k+1] = levelOffsets[k] + levelCount[k];

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

    // ── Diagnostics ───────────────────────────────────────────────────────
    unsigned int thinCount = 0, wideCount = 0;
    for (unsigned int k = 0; k < numLevels; ++k) {
        if (levelCount[k] < MERGE_THRESHOLD) thinCount++;
        else                                  wideCount++;
    }
    // ── Build batch descriptors on host ───────────────────────────────────
    // For each run of consecutive thin levels we build:
    //   batchRows[]    : concatenated row indices
    //   groupOffsets[] : where each original level starts within batchRows
    // These are uploaded to GPU once per batch at launch time.

    const dim3 blockDimWide(TILE_DIM_X, TILE_DIM_Y);
    const unsigned int blockSizeThin = 128; // 1D block for thin kernel

    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

    unsigned int totalLaunches = 0;

    unsigned int k = 0;
    while (k < numLevels) {

        if (levelCount[k] >= MERGE_THRESHOLD) {
            // ── Wide level: launch with 2D grid as before ─────────────────
            dim3 gridDim(
                (numCols       + blockDimWide.x - 1) / blockDimWide.x,
                (levelCount[k] + blockDimWide.y - 1) / blockDimWide.y
            );
            sptrsv_wide_kernel<<<gridDim, blockDimWide, 0, stream>>>(
                L_r, B, X,
                levelRows_d + levelOffsets[k],
                levelCount[k], numCols
            );
            totalLaunches++;
            k++;

        } else {
            // ── Run of thin levels: collect into one batch ────────────────
            unsigned int batchStart = k;

            // Count how many rows this batch will contain
            unsigned int batchSize  = 0;
            unsigned int batchLevels = 0;
            while (k < numLevels && levelCount[k] < MERGE_THRESHOLD) {
                batchSize += levelCount[k];
                batchLevels++;
                k++;
            }

            // Build host-side batchRows and groupOffsets
            unsigned int* batchRows_h    =
                (unsigned int*)malloc(batchSize  * sizeof(unsigned int));
            unsigned int* groupOffsets_h =
                (unsigned int*)malloc((batchLevels + 1) * sizeof(unsigned int));

            unsigned int pos = 0;
            for (unsigned int g = 0; g < batchLevels; ++g) {
                unsigned int lk = batchStart + g;
                groupOffsets_h[g] = pos;
                for (unsigned int r = 0; r < levelCount[lk]; ++r)
                    batchRows_h[pos++] = levelRows[levelOffsets[lk] + r];
            }
            groupOffsets_h[batchLevels] = pos; // sentinel

            // Upload to GPU
            unsigned int* batchRows_d;
            unsigned int* groupOffsets_d;
            CUDA_ERROR_CHECK(cudaMalloc(&batchRows_d,
                             batchSize * sizeof(unsigned int)));
            CUDA_ERROR_CHECK(cudaMalloc(&groupOffsets_d,
                             (batchLevels + 1) * sizeof(unsigned int)));
            CUDA_ERROR_CHECK(cudaMemcpyAsync(batchRows_d, batchRows_h,
                             batchSize * sizeof(unsigned int),
                             cudaMemcpyHostToDevice, stream));
            CUDA_ERROR_CHECK(cudaMemcpyAsync(groupOffsets_d, groupOffsets_h,
                             (batchLevels + 1) * sizeof(unsigned int),
                             cudaMemcpyHostToDevice, stream));

            // ONE kernel launch for the entire batch of thin levels
            unsigned int gridX = (numCols + blockSizeThin - 1) / blockSizeThin;
            sptrsv_thin_batch_kernel<<<gridX, blockSizeThin, 0, stream>>>(
                L_r, B, X,
                batchRows_d, groupOffsets_d,
                batchLevels, batchSize, numCols
            );
            totalLaunches++;

            // Free device temporaries after stream completes
            // (use a host-side deferred free — simple approach: sync then free)
            // For production code, use cudaFreeAsync if available (CUDA 11.2+)
            CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
            CUDA_ERROR_CHECK(cudaFree(batchRows_d));
            CUDA_ERROR_CHECK(cudaFree(groupOffsets_d));

            free(batchRows_h);
            free(groupOffsets_h);
        }
    }

    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

    printf("[gpu2] totalLaunches=%u (was %u), reduction=%.1fx\n",
           totalLaunches, numLevels, (float)numLevels / totalLaunches);

    CUDA_ERROR_CHECK(cudaFree(levelRows_d));
    free(level); free(levelCount); free(levelOffsets);
    free(levelRows); free(fillPos);
}
