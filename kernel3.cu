#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_DIM_X      64
#define TILE_DIM_Y       4
#define MERGE_THRESHOLD 256

// ─────────────────────────────────────────────────────────────────────────────
// OPTIMIZATION 1 (wide kernel): sort rows within each level by decreasing
// row length before uploading levelRows. This improves load balance because
// thread blocks that handle long rows are launched first — by the time short-
// row blocks finish and the GPU has idle SMs, the long-row blocks are still
// running and keep the GPU busy. Without sorting, a single long-row block at
// the end of a level stalls the entire next level.
// ─────────────────────────────────────────────────────────────────────────────

// Wide-level kernel — unchanged from your working gpu2
__global__ void sptrsv_wide_kernel3(
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

// ─────────────────────────────────────────────────────────────────────────────
// OPTIMIZATION 2 (thin nosync kernel): two changes:
//
// A) __ldg() on ALL global reads that are read-only within this kernel:
//    - batchRows[idx]        — read once per row, never written
//    - L_r->colIdxs[j]      — CSR structure, never written
//    - L_r->values[j]       — CSR values, never written
//    - X->values[col*nB+b]  — written by this thread for col==i,
//                              read for col<i (previous rows already done)
//    Using __ldg() routes reads through the read-only L1 texture cache
//    which is separate from the regular L1. For scattered X reads (col
//    jumping around), this significantly improves hit rate.
//
// B) Software prefetch: while computing row i, issue a non-blocking
//    prefetch for the rowPtrs of row i+1. This hides the latency of
//    the next iteration's rowPtr read behind the current row's compute.
//    On Ampere/Volta this maps to an L2 prefetch instruction.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sptrsv_thin_nosync_kernel3(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* batchRows,
        unsigned int  batchSize,
        unsigned int  numCols)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= numCols) return;

    unsigned int nB = numCols;

    // Prefetch first row's metadata before the loop starts
    unsigned int i_cur = __ldg(&batchRows[0]);
    unsigned int rs_cur = __ldg(&L_r->rowPtrs[i_cur]);
    unsigned int re_cur = __ldg(&L_r->rowPtrs[i_cur + 1]);

    for (unsigned int idx = 0; idx < batchSize; ++idx) {
        unsigned int i         = i_cur;
        unsigned int row_start = rs_cur;
        unsigned int row_end   = re_cur;

        // Prefetch next row's metadata while computing current row.
        // __builtin_prefetch hint: locality=1 (L2), rw=0 (read)
        // This is a software hint — the hardware may ignore it but
        // on most NVIDIA GPUs it issues an early load into L2.
        if (idx + 1 < batchSize) {
	    unsigned int i_next  = __ldg(&batchRows[idx + 1]);
            unsigned int rs_next = __ldg(&L_r->rowPtrs[i_next]);

            // PTX prefetch.global.L2 — valid on Volta (V100) and later
            // Issues a hint to bring the cache line into L2 without
            // blocking execution. Unlike __builtin_prefetch this is
            // a genuine device-side instruction.
            const void* ptr = (const void*)(&L_r->colIdxs[rs_next]);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr) : "memory");
        }

        float sum  = __ldg(&B->values[i * nB + b]);
        float diag = 1.0f;

        for (unsigned int j = row_start; j < row_end; ++j) {
            unsigned int col = __ldg(&L_r->colIdxs[j]);
            float        val = __ldg(&L_r->values[j]);

            if (col < i) {
                sum -= val * __ldg(&X->values[col * nB + b]);
            } else if (col == i) {
                diag = (val != 0.0f) ? val : 1.0f;
            }
        }

        X->values[i * nB + b] = sum / diag;
    }
}

// Thin sync kernel — unchanged, used for mixed-size groups
__global__ void sptrsv_thin_sync_kernel3(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* batchRows,
        unsigned int* groupOffsets,
        unsigned int  numGroups,
        unsigned int  batchSize,
        unsigned int  numCols)
{
    unsigned int b      = blockIdx.x * blockDim.x + threadIdx.x;
    bool         active = (b < numCols);
    unsigned int nB     = numCols;

    for (unsigned int g = 0; g < numGroups; ++g) {
        unsigned int gStart = groupOffsets[g];
        unsigned int gEnd   = (g + 1 < numGroups) ? groupOffsets[g + 1] : batchSize;

        if (active) {
            for (unsigned int r = gStart; r < gEnd; ++r) {
                unsigned int i         = batchRows[r];
                unsigned int row_start = L_r->rowPtrs[i];
                unsigned int row_end   = L_r->rowPtrs[i + 1];

                float sum  = B->values[i * nB + b];
                float diag = 1.0f;

                for (unsigned int j = row_start; j < row_end; ++j) {
                    unsigned int col = L_r->colIdxs[j];
                    float        val = L_r->values[j];
                    if      (col < i)  sum -= val * __ldg(&X->values[col * nB + b]);
                    else if (col == i) diag = (val != 0.0f) ? val : 1.0f;
                }

                X->values[i * nB + b] = sum / diag;
            }
        }
        __syncthreads();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row-length comparator for qsort — sorts descending (longest row first)
// ─────────────────────────────────────────────────────────────────────────────
// We pass row lengths via a global pointer since qsort doesn't support
// context. This is preprocessing-only (single-threaded, not performance-
// critical).
static const unsigned int* g_rowPtrs_for_sort = NULL;

static int cmp_row_len_desc(const void* a, const void* b)
{
    unsigned int ra = *(const unsigned int*)a;
    unsigned int rb = *(const unsigned int*)b;
    unsigned int la = g_rowPtrs_for_sort[ra + 1] - g_rowPtrs_for_sort[ra];
    unsigned int lb = g_rowPtrs_for_sort[rb + 1] - g_rowPtrs_for_sort[rb];
    // Descending: longer rows first
    if (lb > la) return  1;
    if (lb < la) return -1;
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain detection
// ─────────────────────────────────────────────────────────────────────────────
static bool detect_chain(CSRMatrix* L_r_host, unsigned int n)
{
    const unsigned int CHECK_ROWS = 2000;
    unsigned int limit = (n < CHECK_ROWS) ? n : CHECK_ROWS;
    for (unsigned int i = 1; i < limit; ++i) {
        unsigned int deps = 0;
        for (unsigned int idx = L_r_host->rowPtrs[i];
                          idx < L_r_host->rowPtrs[i + 1]; ++idx)
            if (L_r_host->colIdxs[idx] < i) deps++;
        if (deps > 1) return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host wrapper
// ─────────────────────────────────────────────────────────────────────────────
void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    // ── Level analysis ────────────────────────────────────────────────────
    unsigned int* level      = (unsigned int*)calloc(n, sizeof(unsigned int));
    unsigned int* levelCount = NULL;
    unsigned int* levelOffsets = NULL;
    unsigned int* levelRows    = NULL;
    unsigned int  numLevels    = 0;

    bool isChain = detect_chain(L_r_host, n);

    if (isChain) {
        numLevels    = n;
        levelCount   = (unsigned int*)malloc(n * sizeof(unsigned int));
        levelOffsets = (unsigned int*)malloc((n + 1) * sizeof(unsigned int));
        levelRows    = (unsigned int*)malloc(n * sizeof(unsigned int));
        for (unsigned int i = 0; i < n; ++i) {
            level[i]        = i;
            levelCount[i]   = 1;
            levelOffsets[i] = i;
            levelRows[i]    = i;
        }
        levelOffsets[n] = n;
    } else {
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
        for (unsigned int i = 0; i < n; ++i)
            if (level[i] > numLevels) numLevels = level[i];
        numLevels++;

        levelCount   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
        levelOffsets = (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
        for (unsigned int i = 0; i < n; ++i) levelCount[level[i]]++;
        levelOffsets[0] = 0;
        for (unsigned int k = 0; k < numLevels; ++k)
            levelOffsets[k + 1] = levelOffsets[k] + levelCount[k];

        levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
        unsigned int* fillPos = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
        for (unsigned int i = 0; i < n; ++i) {
            unsigned int k = level[i];
            levelRows[levelOffsets[k] + fillPos[k]] = i;
            fillPos[k]++;
        }
        free(fillPos);

        // ── OPTIMIZATION 1: sort wide levels by decreasing row length ─────
        // Only sort levels that will use the wide kernel — thin levels use
        // the nosync/sync batch kernel which is order-independent within a
        // group (all size-1 or processed sequentially within a group).
        // Sorting thin levels would break the dependency ordering.
        g_rowPtrs_for_sort = L_r_host->rowPtrs;
        for (unsigned int k = 0; k < numLevels; ++k) {
            if (levelCount[k] >= MERGE_THRESHOLD) {
                qsort(
                    levelRows + levelOffsets[k],
                    levelCount[k],
                    sizeof(unsigned int),
                    cmp_row_len_desc
                );
            }
        }
        g_rowPtrs_for_sort = NULL;
        // ─────────────────────────────────────────────────────────────────
    }

    float avgSize = (float)n / numLevels;
    printf("[gpu2] numLevels=%u  avgSize=%.2f  chain=%s\n",
           numLevels, avgSize, isChain ? "yes" : "no");

    // ── Upload levelRows ──────────────────────────────────────────────────
    unsigned int* levelRows_d;
    CUDA_ERROR_CHECK(cudaMalloc((void**)&levelRows_d, n * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    const dim3         blockDimWide(TILE_DIM_X, TILE_DIM_Y);
    const unsigned int blockSizeThin = 128;

    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

    unsigned int totalLaunches = 0;
    unsigned int k = 0;

    while (k < numLevels) {

        if (levelCount[k] >= MERGE_THRESHOLD) {
            // Wide level — rows already sorted by length
            dim3 gridDim(
                (numCols       + blockDimWide.x - 1) / blockDimWide.x,
                (levelCount[k] + blockDimWide.y - 1) / blockDimWide.y
            );
            sptrsv_wide_kernel3<<<gridDim, blockDimWide, 0, stream>>>(
                L_r, B, X,
                levelRows_d + levelOffsets[k],
                levelCount[k], numCols
            );
            totalLaunches++;
            k++;

        } else {
            // Batch of thin levels
            unsigned int batchStart  = k;
            unsigned int batchSize   = 0;
            unsigned int batchLevels = 0;
            bool         allSizeOne  = true;

            while (k < numLevels && levelCount[k] < MERGE_THRESHOLD) {
                if (levelCount[k] != 1) allSizeOne = false;
                batchSize += levelCount[k];
                batchLevels++;
                k++;
            }

            unsigned int gridX =
                (numCols + blockSizeThin - 1) / blockSizeThin;

            if (allSizeOne) {
                // OPTIMIZATION 2: nosync kernel with __ldg + prefetch
                sptrsv_thin_nosync_kernel3<<<gridX, blockSizeThin, 0, stream>>>(
                    L_r, B, X,
                    levelRows_d + levelOffsets[batchStart],
                    batchSize, numCols
                );
                totalLaunches++;

            } else {
                unsigned int* batchRows_h =
                    (unsigned int*)malloc(batchSize * sizeof(unsigned int));
                unsigned int* groupOffsets_h =
                    (unsigned int*)malloc((batchLevels + 1) * sizeof(unsigned int));

                unsigned int pos = 0;
                for (unsigned int g = 0; g < batchLevels; ++g) {
                    unsigned int lk = batchStart + g;
                    groupOffsets_h[g] = pos;
                    for (unsigned int r = 0; r < levelCount[lk]; ++r)
                        batchRows_h[pos++] = levelRows[levelOffsets[lk] + r];
                }
                groupOffsets_h[batchLevels] = pos;

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

                sptrsv_thin_sync_kernel3<<<gridX, blockSizeThin, 0, stream>>>(
                    L_r, B, X,
                    batchRows_d, groupOffsets_d,
                    batchLevels, batchSize, numCols
                );
                totalLaunches++;

                CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
                CUDA_ERROR_CHECK(cudaFree(batchRows_d));
                CUDA_ERROR_CHECK(cudaFree(groupOffsets_d));
                free(batchRows_h);
                free(groupOffsets_h);
            }
        }
    }

    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

    printf("[gpu2] totalLaunches=%u (numLevels=%u)\n", totalLaunches, numLevels);

    CUDA_ERROR_CHECK(cudaFree(levelRows_d));
    free(level);
    free(levelCount);
    free(levelOffsets);
    free(levelRows);
}

