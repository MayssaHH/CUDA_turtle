// kernel2.cu

// This is another GPU implementation of the sparse triangular solve (SpTRSV).
// This version is still based on the same level-set idea as kernel0 and on the
// same shared-memory tiling idea as kernel1, but here we slightly change how
// the synchronization is handled on the host side.
//
// Main idea:
// - Forward substitution still has row dependencies, so we still cannot solve
//   all rows at the same time.
// - Just like before, we compute dependency levels.
// - Rows in the same level are independent, so they can run in parallel.
// - Also, for one fixed row, many threads solving different RHS columns read
//   the exact same sparse CSR row structure.
// - So we again load the sparse row tiles into shared memory first, and then
//   let all threads for that row reuse the same tile.
//
// So what is different here compared to kernel1?
// - The kernel-side idea is basically the same:
//   - x dimension handles RHS columns
//   - y dimension handles rows inside the same level
//   - shared memory is used to cache CSR tiles for each row
// - The main difference is on the host side scheduling:
//   - in kernel1, after every level launch we explicitly synchronize the device
//   - here, we launch all level kernels on the same CUDA stream
//   - kernels in the same stream execute in order
//   - so correctness is still preserved, but we only do one stream
//     synchronization at the very end
//
// Overall flow of this file:
// 1. On the host, compute the dependency level of each row.
// 2. Group rows by level.
// 3. Copy the compact levelRows array to the GPU.
// 4. For each level, launch one kernel on the same CUDA stream.
// 5. Inside the kernel, use shared memory tiling to reduce repeated reads of
//    the sparse row structure.
// 6. Synchronize once at the end after all level kernels have been queued.

#include "common.h"

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
    // Thread mapping:
    // - x dimension handles RHS columns
    // - y dimension handles rows inside the current level
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard against out-of-bounds threads
    if (b >= numCols || r >= levelSize) return;

    // Actual row in the matrix corresponding to this thread
    unsigned int i  = levelRows[r];
    unsigned int nB = numCols;

    // CSR range for row i
    unsigned int row_start = L_r->rowPtrs[i];
    unsigned int row_end   = L_r->rowPtrs[i + 1];
    unsigned int row_len   = row_end - row_start;

    // Each matrix row handled by a different threadIdx.y gets its own shared
    // memory slice.
    // So for one block:
    // - s_col[ty][tx] stores column indices of one tile for row ty
    // - s_val[ty][tx] stores the matching values of that tile
    // This avoids any mixing between different rows in the block.
    __shared__ unsigned int s_col[TILE_DIM_Y][TILE_DIM_X];
    __shared__ float        s_val[TILE_DIM_Y][TILE_DIM_X];

    // Start from the RHS value B(i,b)
    float sum  = B->values[i * nB + b];

    // This will hold the diagonal entry L(i,i)
    float diag = 1.0f;

    // Traverse row i tile by tile
    for (unsigned int base = 0; base < row_len; base += TILE_DIM_X) {

        // Step 1: cooperative load
        // All x-threads for this row load one sparse entry each into shared
        // memory, so later all of them can reuse the same cached tile
        unsigned int k = base + threadIdx.x;
        if (k < row_len) {
            s_col[threadIdx.y][threadIdx.x] = L_r->colIdxs[row_start + k];
            s_val[threadIdx.y][threadIdx.x] = L_r->values[row_start + k];
        } else {
            // If the tile goes past the end of the row, pad with a harmless
            // sentinel.
            // col == i is safe here because the matching value is 0, so it
            // does not change sum and does not affect diag in practice
            s_col[threadIdx.y][threadIdx.x] = i;
            s_val[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Make sure the full tile is loaded before any thread uses it
        __syncthreads();

        // Step 2: process the current tile for this row
        unsigned int tile_limit = min(TILE_DIM_X, row_len - base);

        for (unsigned int j = 0; j < tile_limit; ++j) {
            unsigned int col = s_col[threadIdx.y][j];
            float        val = s_val[threadIdx.y][j];

            // If col < i, this is a dependency contribution coming from a row
            // that should already be solved
            if (col < i) {
                sum -= val * X->values[col * nB + b];

            // If col == i, this is the diagonal entry needed for the final divide
            } else if (col == i) {
                diag = val;
                if (diag == 0.0f) {
                    diag = 1.0f;
                }
            }
        }

        // Make sure no thread is still reading this tile before we overwrite it
        __syncthreads();
    }

    // Final forward substitution result for X(i,b)
    X->values[i * nB + b] = sum / diag;
}

// Host wrapper
void sptrsv_gpu2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                    CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    // Number of rows in the triangular system
    unsigned int n = L_r_host->numRows;

    // Step 1: dependency analysis
    // level[i] = longest dependency chain ending at row i
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));

    // Since the matrix is lower triangular, whenever we process row i, all rows
    // col < i have already been considered in this host-side pass
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i];
                          idx < L_r_host->rowPtrs[i + 1]; ++idx) {

            unsigned int col = L_r_host->colIdxs[idx];

            if (col < i) {
                // If row i depends on row col, then i must be placed at least
                // one level after col
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
    numLevels++; // levels start from 0

    // Step 3: count how many rows belong to each level
    unsigned int* levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        levelCount[level[i]]++;
    }

    // Step 4: prefix sum over level counts
    // This tells us where each level starts in the compact levelRows array
    unsigned int* levelOffsets =
        (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));

    levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k) {
        levelOffsets[k + 1] = levelOffsets[k] + levelCount[k];
    }

    // Step 5: build levelRows
    // This stores the actual row indices grouped level by level
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
    // - x dimension handles RHS columns
    // - y dimension handles rows inside the current level
    const dim3 blockDim(64, 4);

    // All level kernels are queued on the same CUDA stream.
    // Since kernels in the same stream execute in order, level k+1 cannot start
    // before level k, so the dependency ordering is still correct even without
    // calling cudaDeviceSynchronize after every level
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Step 7: process the matrix level by level
    // The launches happen in order on the same stream
    for (unsigned int k = 0; k < numLevels; ++k) {

        unsigned int levelSize  = levelCount[k];
        unsigned int levelStart = levelOffsets[k];

        // Grid covers all (row, column) pairs in this level
        dim3 gridDim(
            (numCols   + blockDim.x - 1) / blockDim.x,
            (levelSize + blockDim.y - 1) / blockDim.y
        );

        // Launch the kernel for the current level on the same stream
        sptrsv_gpu2_kernel<<<gridDim, blockDim, 0, stream>>>(
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

    // Single synchronization at the end:
    // all queued level kernels must finish before we continue
    cudaStreamSynchronize(stream);

    // Cleanup
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    CUDA_ERROR_CHECK(cudaFree(levelRows_d));

    free(level);
    free(levelCount);
    free(levelOffsets);
    free(levelRows);
    free(fillPos);
}
