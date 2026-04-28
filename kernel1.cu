// kernel1.cu

// This is another GPU implementation of the sparse triangular solve (SpTRSV).
// This version is based on the level-set version from kernel0, but now we try
// to optimize the sparse inner loop itself using shared memory.
//
// Main idea:
// - In the level-set version, rows in the same level are independent, so they
//   can be solved in parallel.
// - But inside each row solve, every thread working on a different RHS column
//   still reads the same sparse row structure again and again from global
//   memory.
// - More specifically, for one fixed row i, all x-threads of that row walk
//   through exactly the same CSR entries:
//      L_r->colIdxs[row_start ... row_end-1]
//      L_r->values[row_start ... row_end-1]
// - So there is a lot of redundant global memory traffic there.
//
// What we do in this file:
// - We keep the same level-set scheduling idea from kernel0.
// - We still parallelize:
//   - across RHS columns in x
//   - across rows in the same level in y
// - But now, for each row, the x-threads cooperate to load a tile of the CSR
//   row into shared memory first.
// - Then all those threads reuse that shared tile instead of rereading the
//   same sparse entries from global memory many times.
//
// Important detail:
// - B and X stay in the normal row-major layout here.
// - So this version is only changing how the sparse row of L is accessed.
// - We are not doing any transpose or layout conversion in this file.
//
// So compared to kernel0:
// - kernel0 already improved parallelism using levels
// - this version keeps that idea
// - but adds shared memory tiling to reduce repeated reads of CSR structure
//
// Block structure used here:
// - TILE_DIM_X = 64 threads for RHS columns
// - TILE_DIM_Y = 4 rows from the same level per block
// - each y-row inside the block gets its own shared-memory slice

#include "common.h"

#define TILE_DIM_X 64
#define TILE_DIM_Y 4


__global__ void sptrsv_gpu1_kernel(
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
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard against out-of-bounds threads
    if (b >= numCols || r >= levelSize) return;

    // Actual matrix row for this thread
    unsigned int i  = levelRows[r];
    unsigned int nB = numCols;

    // CSR range for row i
    unsigned int row_start = L_r->rowPtrs[i];
    unsigned int row_end   = L_r->rowPtrs[i + 1];
    unsigned int row_len   = row_end - row_start;

    // Shared memory buffers:
    // For each ty row in the block, we store one tile of:
    // - column indices
    // - values
    // This way different matrix rows do not overwrite each other's tile
    __shared__ unsigned int s_col[TILE_DIM_Y][TILE_DIM_X];
    __shared__ float        s_val[TILE_DIM_Y][TILE_DIM_X];

    // Start from the RHS value B(i,b)
    float sum  = B->values[i * nB + b];

    // This will hold the diagonal entry L(i,i)
    float diag = 1.0f;

    // Traverse row i tile by tile
    for (unsigned int base = 0; base < row_len; base += TILE_DIM_X) {

        // Cooperative load:
        // all x-threads for this row load one sparse entry each into shared
        // memory, so later all of them can reuse the same tile
        unsigned int k = base + threadIdx.x;
        if (k < row_len) {
            s_col[threadIdx.y][threadIdx.x] = L_r->colIdxs[row_start + k];
            s_val[threadIdx.y][threadIdx.x] = L_r->values[row_start + k];
        } else {
            // Pad the tile if the row ends before the tile ends
            // Using col == i as a sentinel is safe here:
            // it does not add to sum, and the value is 0 anyway
            s_col[threadIdx.y][threadIdx.x] = i;
            s_val[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Make sure the whole tile is visible before any thread starts using it
        __syncthreads();

        // Number of valid sparse entries inside this tile
        unsigned int tile_limit = min(TILE_DIM_X, row_len - base);

        // Now every x-thread solving this row reuses the same shared tile
        for (unsigned int j = 0; j < tile_limit; ++j) {
            unsigned int col = s_col[threadIdx.y][j];
            float        val = s_val[threadIdx.y][j];

            // If col < i, this is a previously solved dependency contribution
            if (col < i) {
                sum -= val * X->values[col * nB + b];

            // If col == i, this is the diagonal element needed at the end
            } else if (col == i) {
                diag = (val != 0.0f) ? val : 1.0f;
            }
        }

        // Make sure no thread is still using this tile before we overwrite it
        __syncthreads();
    }

    // Final forward substitution result for X(i,b)
    X->values[i * nB + b] = sum / diag;
}


void sptrsv_gpu1(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    // These inputs are not needed here
    // This version still relies on CSR for dependency analysis and for the row
    // traversal inside the kernel
    (void)L_c;
    (void)L_c_host;

    unsigned int n = L_r_host->numRows;

    // Step 1: level-set analysis
    // level[i] = longest dependency chain ending at row i
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i]; idx < L_r_host->rowPtrs[i + 1]; ++idx) {
            unsigned int col = L_r_host->colIdxs[idx];
            if (col < i) {
                // If row i depends on row col, then i must come after col
                unsigned int candidate = level[col] + 1;
                if (candidate > level[i]) level[i] = candidate;
            }
        }
    }

    // Step 2: compute total number of levels
    unsigned int numLevels = 0;
    for (unsigned int i = 0; i < n; ++i)
        if (level[i] > numLevels) numLevels = level[i];
    numLevels++;

    // Step 3: count rows per level
    unsigned int* levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i)
        levelCount[level[i]]++;

    // Step 4: prefix sum to know where each level begins inside levelRows
    unsigned int* levelOffsets = (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
    levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k)
        levelOffsets[k + 1] = levelOffsets[k] + levelCount[k];

    // Step 5: build levelRows
    // This array stores actual row indices grouped by level
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
    // - y dimension for rows inside the same level
    const dim3 blockDim(TILE_DIM_X, TILE_DIM_Y);

    // Step 6: solve the matrix one level at a time
    // Rows in the same level are independent, but different levels still have
    // dependencies, so we must synchronize between levels
    for (unsigned int k = 0; k < numLevels; ++k) {
        unsigned int levelSize  = levelCount[k];
        unsigned int levelStart = levelOffsets[k];

        // Grid covers all (row, column) pairs in this level
        dim3 gridDim(
            (numCols   + blockDim.x - 1) / blockDim.x,
            (levelSize + blockDim.y - 1) / blockDim.y
        );

        // Launch current level
        sptrsv_gpu1_kernel<<<gridDim, blockDim>>>(
            L_r, B, X,
            levelRows_d + levelStart,
            levelSize, numCols
        );

        // Synchronization is necessary because the next level may read values
        // produced by this level
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
