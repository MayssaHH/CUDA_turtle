// kernel1.cu
//
// First Optimization over kernel0_v2: column-major dense matrix layout.
//
// In kernel0_v2, B and X are row-major: X(j, b) lives at j*numCols + b.
// The sparse inner loop reads X(j, b) for irregular j values, jumping by
// strides of numCols between accesses -- each cache line fetch uses only
// 1 of 32 floats (~3% utilization).
//
// Fix: store B and X in column-major order: X(j, b) lives at b*numRows + j.
// All values a thread ever reads are contiguous in memory, so cache lines
// fetched for one j are reused for nearby j values in the same sparse row.
//
// The level-set scheduling, grid/block dimensions, and launch loop are
// identical to kernel0_v2. Only the index arithmetic changes in support of the first optimization.

#include "common.h"


// Each thread owns one (row i, RHS column b) pair.
// x-dim of the grid covers RHS columns, y-dim covers rows within the current level.
// B and X are expected in column-major layout when this kernel runs.
__global__ void sptrsv_gpu1_kernel(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        unsigned int* levelRows,   // row indices for the current level (device ptr, pre-offset)
        unsigned int  levelSize,   // number of rows in this level
        unsigned int  numRows,     // total matrix rows, needed for column-major addressing
        unsigned int  numCols)     // number of RHS columns
{
    unsigned int b        = blockIdx.x * blockDim.x + threadIdx.x;  // RHS column index
    unsigned int levelIdx = blockIdx.y * blockDim.y + threadIdx.y;  // index within level

    // threads at the edge of the grid that fall outside the valid range exit early
    if (b >= numCols || levelIdx >= levelSize) return;

    // translate within-level index to the actual matrix row
    unsigned int i = levelRows[levelIdx];

    // column-major read: column b, row i -> address b*numRows + i
    // (in row-major this was i*numCols + b, which strided across memory as i changed)
    float sum  = B->values[b * numRows + i];
    float diag = 1.0f;  // default diagonal; overwritten when we encounter L(i,i)

    // sparse inner loop over nonzeros in row i of L
    for (unsigned int idx = L_r->rowPtrs[i];
                      idx < L_r->rowPtrs[i + 1]; ++idx) {

        unsigned int col = L_r->colIdxs[idx];
        float        val = L_r->values[idx];

        if (col < i) {
            // off-diagonal: subtract contribution of already-solved row col
            // column-major read: b*numRows + col -- stays in the same contiguous
            // block of memory as all other reads this thread makes, unlike row-major
            sum -= val * X->values[b * numRows + col];

        } else if (col == i) {
            // diagonal: record L(i,i); guard against explicit zero just in case
            diag = (val != 0.0f) ? val : 1.0f;
        }
    }

    // write solution X(i,b) in column-major layout
    X->values[b * numRows + i] = sum / diag;
}



// Transpose helpers (host-side, called once before and after the kernel loop)
// row-major src -> column-major dst: dst[j*numRows + i] = src[i*numCols + j]
static void transposeToColMajor(const float* src, float* dst,
                                 unsigned int numRows, unsigned int numCols)
{
    for (unsigned int i = 0; i < numRows; ++i)
        for (unsigned int j = 0; j < numCols; ++j)
            dst[j * numRows + i] = src[i * numCols + j];
}

// column-major src -> row-major dst: inverse of transposeToColMajor
static void transposeToRowMajor(const float* src, float* dst,
                                 unsigned int numRows, unsigned int numCols)
{
    for (unsigned int j = 0; j < numCols; ++j)
        for (unsigned int i = 0; i < numRows; ++i)
            dst[i * numCols + j] = src[j * numRows + i];
}



// Host wrapper
void sptrsv_gpu1(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int numRows = L_r_host->numRows;
    unsigned int n       = numRows;
    size_t       sz      = (size_t)numRows * numCols * sizeof(float);

    // the DenseMatrix struct itself lives on the device, so we download the header
    // to get the device-side values pointer before touching the float data
    DenseMatrix B_hdr, X_hdr;
    CUDA_ERROR_CHECK(cudaMemcpy(&B_hdr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(&X_hdr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));

    // reusable host scratch buffer to avoid repeated malloc for large matrices
    float* tmp = (float*)malloc(sz);


    // Step 0: transpose B on device from row-major to column-major
    // download B, rearrange in place on the host, upload back to the same device buffer
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, B_hdr.values, sz, cudaMemcpyDeviceToHost));
    float* B_cm = (float*)malloc(sz);
    transposeToColMajor(tmp, B_cm, numRows, numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(B_hdr.values, B_cm, sz, cudaMemcpyHostToDevice));
    free(B_cm);


    // Level-set preprocessing (identical to kernel0_v2)
    // level[i] = length of the longest dependency chain ending at row i
    // computed in one forward pass since L is lower-triangular
    unsigned int* level = (unsigned int*)calloc(n, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i];
                          idx < L_r_host->rowPtrs[i + 1]; ++idx) {
            unsigned int col = L_r_host->colIdxs[idx];
            if (col < i) {
                // row i depends on col, so it must sit at a strictly deeper level
                unsigned int candidate = level[col] + 1;
                if (candidate > level[i]) level[i] = candidate;
            }
        }
    }

    // levels are 0-indexed, so total count is max level + 1
    unsigned int numLevels = 0;
    for (unsigned int i = 0; i < n; ++i)
        if (level[i] > numLevels) numLevels = level[i];
    numLevels++;

    // count rows per level to size each kernel launch
    unsigned int* levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i)
        levelCount[level[i]]++;

    // prefix sum: levelOffsets[k] = start index of level k inside levelRows[]
    unsigned int* levelOffsets = (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
    levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k)
        levelOffsets[k + 1] = levelOffsets[k] + levelCount[k];

    // pack row indices grouped by level; fillPos[k] is the write cursor for level k
    unsigned int* levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* fillPos   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int k = level[i];
        levelRows[levelOffsets[k] + fillPos[k]] = i;
        fillPos[k]++;
    }

    // upload the full levelRows array once; kernel launches index into it via pointer offsets
    unsigned int* levelRows_d;
    CUDA_ERROR_CHECK(cudaMalloc((void**)&levelRows_d, n * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));


    // Kernel launch loop (identical structure to kernel0_v2)
    // fixed 16x16 block: x covers RHS columns, y covers rows within the level
    const dim3 blockDim(16, 16);

    for (unsigned int k = 0; k < numLevels; ++k) {
        unsigned int levelSize  = levelCount[k];
        unsigned int levelStart = levelOffsets[k];

        // grid y-dim adapts to how many rows are in this level
        dim3 gridDim(
            (numCols   + blockDim.x - 1) / blockDim.x,
            (levelSize + blockDim.y - 1) / blockDim.y
        );

        // pass a pointer offset into levelRows_d so the kernel sees indices [0, levelSize)
        sptrsv_gpu1_kernel<<<gridDim, blockDim>>>(
            L_r, B, X,
            levelRows_d + levelStart,
            levelSize, numRows, numCols
        );

        // barrier: all X writes from level k must be globally visible before level k+1 reads them
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }


    // Transpose X back to row-major and restore B to row-major
    // X is currently column-major on the device; verify() in main.cu expects row-major
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, X_hdr.values, sz, cudaMemcpyDeviceToHost));
    float* X_rm = (float*)malloc(sz);
    transposeToRowMajor(tmp, X_rm, numRows, numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(X_hdr.values, X_rm, sz, cudaMemcpyHostToDevice));
    free(X_rm);

    // restore B to row-major so the caller's buffer is left in its original state
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, B_hdr.values, sz, cudaMemcpyDeviceToHost));
    float* B_rm = (float*)malloc(sz);
    transposeToRowMajor(tmp, B_rm, numRows, numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(B_hdr.values, B_rm, sz, cudaMemcpyHostToDevice));
    free(B_rm);

    free(tmp);


    // Cleanup
    CUDA_ERROR_CHECK(cudaFree(levelRows_d));
    free(level);
    free(levelCount);
    free(levelOffsets);
    free(levelRows);
    free(fillPos);
}
