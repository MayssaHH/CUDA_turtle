// kernel3.cu

// This is another GPU implementation of the sparse triangular solve (SpTRSV).
// This version is mainly targeting the case where the number of RHS columns is
// large, for example 512 columns, because in that case we want each thread to
// do more useful work after each sparse tile load.
//
// Main idea:
// - We still use the same level-set scheduling idea as kernel0, kernel1, and
//   kernel2.
// - So rows in the same level are independent and can run in parallel.
// - We also still use shared memory tiling for the sparse CSR row structure,
//   because threads solving different RHS columns for the same row all need the
//   same sparse entries.
// - But now we add one more optimization on top of that:
//   each x-thread does not solve only one RHS column anymore.
// - Instead, each thread owns REG_COLS columns and keeps their partial sums in
//   registers.
//
// Why this helps:
// - In the shared-memory tiled versions, after loading one sparse tile, each
//   thread uses that tile to update only one column.
// - Here, after loading the same sparse tile, each thread updates multiple RHS
//   columns.
// - So we get more arithmetic work per shared-memory tile load and per
//   synchronization point.
// - This is especially useful when numCols is large.
//
// Overall structure of this file:
// 1. On the host, compute dependency levels exactly like the level-set versions.
// 2. Group rows by level.
// 3. Launch one kernel per level on the same CUDA stream.
// 4. Inside the kernel:
//    - y dimension still chooses which row in the level we are solving
//    - x dimension still spreads work across RHS columns
//    - but now each x-thread owns REG_COLS different columns
//    - sparse CSR tiles are still cached in shared memory
//    - the multiple RHS sums are kept in registers
//
// So compared to kernel2:
// - kernel2 already had:
//   - level-set scheduling
//   - shared-memory tiling for CSR row data
//   - one stream-level synchronization at the end
// - this version keeps all of that
// - but adds register blocking across RHS columns

#include "common.h"

#define TILE_DIM_Y 4
#define TILE_DIM_X 64
#define REG_COLS   4


__global__ void sptrsv_gpu3_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,
        unsigned int  levelSize,
        unsigned int  numCols)
{
    // Shared memory used for one sparse tile per row handled in the block
    // shm_nnz stores how many nonzeros each threadIdx.y row has, so all threads
    // in the block can agree on how many tiles the block must iterate through
    __shared__ unsigned int shm_nnz[TILE_DIM_Y];
    __shared__ unsigned int s_col[TILE_DIM_Y][TILE_DIM_X];
    __shared__ float        s_val[TILE_DIM_Y][TILE_DIM_X];

    // Thread mapping:
    // - tx handles column positions inside the block
    // - ty selects which matrix row inside the current level this thread belongs to
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int levelIdx = blockIdx.y * TILE_DIM_Y + ty;

    // Some threads in the last block row may not correspond to a real row
    bool row_valid = (levelIdx < levelSize);

    // Actual matrix row handled by this thread
    unsigned int i = row_valid ? levelRows[levelIdx] : 0u;
    unsigned int rowStart = row_valid ? L_r->rowPtrs[i] : 0u;
    unsigned int rowEnd   = row_valid ? L_r->rowPtrs[i + 1] : 0u;
    unsigned int nnz      = rowEnd - rowStart;

    // Only one x-thread per row writes the row length into shared memory
    if (tx == 0) {
        shm_nnz[ty] = nnz;
    }
    __syncthreads();

    // We take the maximum nnz among all rows in this block so every thread
    // executes the same number of tile iterations and reaches the same
    // __syncthreads calls safely
    unsigned int maxNnz = 0;
    for (unsigned int r = 0; r < TILE_DIM_Y; ++r) {
        if (shm_nnz[r] > maxNnz) {
            maxNnz = shm_nnz[r];
        }
    }

    // Each thread owns REG_COLS different RHS columns
    // For example if REG_COLS = 4, one thread updates 4 independent sums
    unsigned int baseCol = blockIdx.x * (TILE_DIM_X * REG_COLS) + tx;
    unsigned int cols[REG_COLS];
    float sums[REG_COLS];
    bool active[REG_COLS];

    // Initialize the owned columns and their starting sums from B
    // active[c] tells us whether that register-column is actually inside
    // the valid numCols range
#pragma unroll
    for (unsigned int c = 0; c < REG_COLS; ++c) {
        cols[c] = baseCol + c * TILE_DIM_X;
        active[c] = row_valid && (cols[c] < numCols);
        sums[c] = active[c] ? B->values[i * numCols + cols[c]] : 0.0f;
    }

    // This will hold the diagonal entry L(i,i)
    float diag = 1.0f;

    // Total number of sparse tiles needed by the block
    unsigned int numTiles = (maxNnz + TILE_DIM_X - 1) / TILE_DIM_X;

    // Traverse the sparse row tile by tile
    for (unsigned int tile = 0; tile < numTiles; ++tile) {
        unsigned int localIdx = tile * TILE_DIM_X + tx;

        // Cooperative load:
        // each x-thread loads one sparse entry for its own row into shared memory
        if (row_valid && localIdx < nnz) {
            s_col[ty][tx] = L_r->colIdxs[rowStart + localIdx];
            s_val[ty][tx] = L_r->values[rowStart + localIdx];
        } else {
            // Pad with a harmless sentinel for rows that have already exhausted
            // their real nonzeros
            s_col[ty][tx] = 0xFFFFFFFFu;
            s_val[ty][tx] = 0.0f;
        }

        // Make sure the whole tile is ready before using it
        __syncthreads();

        if (row_valid) {
            // Number of valid sparse entries for this row in this specific tile
            unsigned int tileOffset = tile * TILE_DIM_X;
            unsigned int remaining = (nnz > tileOffset) ? (nnz - tileOffset) : 0u;
            unsigned int tileLen = (remaining < (unsigned int)TILE_DIM_X)
                                   ? remaining
                                   : (unsigned int)TILE_DIM_X;

            // Reuse the same sparse tile for all REG_COLS sums owned by the thread
            for (unsigned int j = 0; j < tileLen; ++j) {
                unsigned int col = s_col[ty][j];
                float val = s_val[ty][j];

                // If col < i, this is a dependency contribution from a previously
                // solved row, so subtract it from every active register sum
                if (col < i) {
#pragma unroll
                    for (unsigned int c = 0; c < REG_COLS; ++c) {
                        if (active[c]) {
                            sums[c] -= val * X->values[col * numCols + cols[c]];
                        }
                    }

                // If col == i, this is the diagonal entry needed for the final divide
                } else if (col == i) {
                    diag = (val != 0.0f) ? val : 1.0f;
                }
            }
        }

        // Make sure no thread is still reading the tile before overwriting it
        __syncthreads();
    }

    // Write all valid register-held results back to X
    if (row_valid) {
#pragma unroll
        for (unsigned int c = 0; c < REG_COLS; ++c) {
            if (active[c]) {
                X->values[i * numCols + cols[c]] = sums[c] / diag;
            }
        }
    }
}


void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    // These inputs are not needed here
    // This version still relies on CSR for both dependency analysis and the
    // actual row traversal
    (void)L_c;
    (void)L_c_host;

    unsigned int n = L_r_host->numRows;
    if (n == 0 || numCols == 0) return;

    // Step 1: dependency analysis
    // level[i] = longest dependency chain ending at row i
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i]; idx < L_r_host->rowPtrs[i + 1]; ++idx) {
            unsigned int col = L_r_host->colIdxs[idx];
            if (col < i) {
                // If row i depends on row col, then row i must come after col
                unsigned int candidate = level[col] + 1;
                if (candidate > level[i]) level[i] = candidate;
            }
        }
    }

    // Step 2: determine total number of levels
    unsigned int numLevels = 0;
    for (unsigned int i = 0; i < n; ++i) {
        if (level[i] > numLevels) numLevels = level[i];
    }
    numLevels++;

    // Step 3: count rows per level
    unsigned int* levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        levelCount[level[i]]++;
    }

    // Step 4: prefix sum over level counts
    // This tells us where each level starts inside the compact levelRows array
    unsigned int* levelOffsets =
        (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
    levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k) {
        levelOffsets[k + 1] = levelOffsets[k] + levelCount[k];
    }

    // Step 5: build levelRows
    // This stores actual row indices grouped level by level
    unsigned int* levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* fillPos   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int k = level[i];
        levelRows[levelOffsets[k] + fillPos[k]] = i;
        fillPos[k]++;
    }

    // Step 6: copy levelRows to the GPU
    unsigned int* levelRows_d;
    CUDA_ERROR_CHECK(cudaMalloc((void**)&levelRows_d, n * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    // 2D block:
    // - x dimension still uses TILE_DIM_X threads
    // - but each x-thread now covers REG_COLS columns
    // - y dimension covers rows inside the same level
    const dim3 blockDim(TILE_DIM_X, TILE_DIM_Y);

    // One block covers this many RHS columns total
    const unsigned int blockColSpan = TILE_DIM_X * REG_COLS;

    // Same stream-ordering idea as kernel2:
    // all level kernels are queued on the same stream, so execution order is
    // preserved without synchronizing after every level
    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

    // Step 7: process the matrix level by level
    for (unsigned int k = 0; k < numLevels; ++k) {
        unsigned int levelSize  = levelCount[k];
        unsigned int levelStart = levelOffsets[k];

        // Grid covers all (row, column-block) pairs in this level
        dim3 gridDim(
            (numCols   + blockColSpan - 1) / blockColSpan,
            (levelSize + blockDim.y - 1) / blockDim.y
        );

        // Launch the current level
        sptrsv_gpu3_kernel<<<gridDim, blockDim, 0, stream>>>(
            L_r,
            B,
            X,
            levelRows_d + levelStart,
            levelSize,
            numCols
        );

        // Check launch errors immediately
        CUDA_ERROR_CHECK(cudaGetLastError());
    }

    // Single synchronization at the end after all level kernels are queued
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
    CUDA_ERROR_CHECK(cudaFree(levelRows_d));

    free(level);
    free(levelCount);
    free(levelOffsets);
    free(levelRows);
    free(fillPos);
}
