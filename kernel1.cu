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


// Static state shared between preprocess, solve, and postprocess.
// Safe because only one kernel1 run is active at a time.
static CSRMatrix*    s_L_r         = NULL;  // device ptr to CSR matrix struct
static DenseMatrix*  s_B           = NULL;  // device ptr to B struct
static DenseMatrix*  s_X           = NULL;  // device ptr to X struct
static float*        s_B_dev_vals  = NULL;  // device ptr to B's float data
static float*        s_X_dev_vals  = NULL;  // device ptr to X's float data
static unsigned int* s_levelRows_d = NULL;  // device array of row indices grouped by level
static unsigned int* s_levelCount  = NULL;  // host array: number of rows per level
static unsigned int* s_levelOffsets= NULL;  // host array: start offset of each level in levelRows
static unsigned int  s_numLevels   = 0;
static unsigned int  s_numRows     = 0;
static unsigned int  s_numCols     = 0;


// sptrsv_gpu1_preprocess
// Performs all setup that does NOT belong in the timed region:
//   - transposes B to column-major on the device
//   - computes the level sets from the sparsity pattern of L
//   - uploads the levelRows array to the device
// Results are stored in the static state above for solve() to consume.
void sptrsv_gpu1_preprocess(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                             CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int numRows = L_r_host->numRows;
    unsigned int n       = numRows;
    size_t       sz      = (size_t)numRows * numCols * sizeof(float);

    // store pointers so solve() and postprocess() can reference them without extra args
    s_L_r    = L_r;
    s_B      = B;
    s_X      = X;
    s_numRows = numRows;
    s_numCols = numCols;

    // download struct headers to extract the device-side float data pointers
    DenseMatrix B_hdr, X_hdr;
    CUDA_ERROR_CHECK(cudaMemcpy(&B_hdr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(&X_hdr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost));
    s_B_dev_vals = B_hdr.values;  // device float* for B's data
    s_X_dev_vals = X_hdr.values;  // device float* for X's data

    // transpose B from row-major to column-major on the device
    float* tmp = (float*)malloc(sz);
    float* B_cm = (float*)malloc(sz);
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, s_B_dev_vals, sz, cudaMemcpyDeviceToHost));
    transposeToColMajor(tmp, B_cm, numRows, numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(s_B_dev_vals, B_cm, sz, cudaMemcpyHostToDevice));
    free(B_cm);
    free(tmp);

    // level-set analysis -- identical to kernel0_v2
    // level[i] = length of the longest dependency chain ending at row i
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
    s_numLevels = numLevels;

    // count rows per level to size each kernel launch
    s_levelCount = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i)
        s_levelCount[level[i]]++;

    // prefix sum: s_levelOffsets[k] = start index of level k inside levelRows[]
    s_levelOffsets = (unsigned int*)malloc((numLevels + 1) * sizeof(unsigned int));
    s_levelOffsets[0] = 0;
    for (unsigned int k = 0; k < numLevels; ++k)
        s_levelOffsets[k + 1] = s_levelOffsets[k] + s_levelCount[k];

    // pack row indices grouped by level; fillPos[k] is the write cursor for level k
    unsigned int* levelRows = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* fillPos   = (unsigned int*)calloc(numLevels, sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int k = level[i];
        levelRows[s_levelOffsets[k] + fillPos[k]] = i;
        fillPos[k]++;
    }

    // upload the full levelRows array once; solve() indexes into it via pointer offsets
    CUDA_ERROR_CHECK(cudaMalloc((void**)&s_levelRows_d, n * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemcpy(s_levelRows_d, levelRows, n * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));

    free(level);
    free(levelRows);
    free(fillPos);
}


// sptrsv_gpu1_solve
// The timed region: launches the level-set kernels and nothing else.
// All setup was done in preprocess(); all cleanup is in postprocess().
void sptrsv_gpu1_solve()
{
    // fixed 16x16 block: x covers RHS columns, y covers rows within the level
    const dim3 blockDim(16, 16);

    for (unsigned int k = 0; k < s_numLevels; ++k) {
        unsigned int levelSize  = s_levelCount[k];
        unsigned int levelStart = s_levelOffsets[k];

        // grid y-dim adapts to how many rows are in this level
        dim3 gridDim(
            (s_numCols   + blockDim.x - 1) / blockDim.x,
            (levelSize   + blockDim.y - 1) / blockDim.y
        );

        // pass a pointer offset into s_levelRows_d so the kernel sees indices [0, levelSize)
        sptrsv_gpu1_kernel<<<gridDim, blockDim>>>(
            s_L_r, s_B, s_X,
            s_levelRows_d + levelStart,
            levelSize, s_numRows, s_numCols
        );

        // barrier: all X writes from level k must be globally visible before level k+1 reads them
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
}


// sptrsv_gpu1_postprocess
// Performs all teardown that does NOT belong in the timed region:
//   - transposes X back to row-major so verify() in main.cu sees the right layout
//   - restores B to row-major so the caller's buffer is left in its original state
//   - frees all precomputed state
void sptrsv_gpu1_postprocess()
{
    size_t sz = (size_t)s_numRows * s_numCols * sizeof(float);
    float* tmp = (float*)malloc(sz);
    float* buf = (float*)malloc(sz);

    // X is column-major on device; verify() expects row-major
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, s_X_dev_vals, sz, cudaMemcpyDeviceToHost));
    transposeToRowMajor(tmp, buf, s_numRows, s_numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(s_X_dev_vals, buf, sz, cudaMemcpyHostToDevice));

    // restore B to row-major so the caller's device buffer is unchanged
    CUDA_ERROR_CHECK(cudaMemcpy(tmp, s_B_dev_vals, sz, cudaMemcpyDeviceToHost));
    transposeToRowMajor(tmp, buf, s_numRows, s_numCols);
    CUDA_ERROR_CHECK(cudaMemcpy(s_B_dev_vals, buf, sz, cudaMemcpyHostToDevice));

    free(tmp);
    free(buf);

    // free level-set state
    CUDA_ERROR_CHECK(cudaFree(s_levelRows_d));
    free(s_levelCount);
    free(s_levelOffsets);

    // clear statics so a subsequent call starts fresh
    s_L_r = NULL; s_B = NULL; s_X = NULL;
    s_B_dev_vals = NULL; s_X_dev_vals = NULL;
    s_levelRows_d = NULL; s_levelCount = NULL; s_levelOffsets = NULL;
    s_numLevels = 0; s_numRows = 0; s_numCols = 0;
}
