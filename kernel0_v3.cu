// kernel0_v3.cu

#include "common.h"

// This is the third GPU version of SpTRSV.
// In this version we move away from explicit level by level execution
// and instead use a dynamic scheduling approach based on dependencies.
//
// Idea:
// Each row keeps track of how many dependencies are still unresolved.
// A row becomes ready once all rows it depends on are completed.
// Blocks repeatedly pick a ready row, compute it, then update its dependents.
//
// Compared to v2:
// v2 had strict level by level synchronization.
// Here we allow rows to be processed as soon as they become ready.
// This removes global barriers between levels and increases parallelism.

__global__ void sptrsv_gpu0_kernel_v3(
        CSRMatrix*    L_r,
        DenseMatrix*  B,
        DenseMatrix*  X,
        int*          depCounter,
        int*          rowReady,
        unsigned int* dependents,
        unsigned int* dependentOffsets,
        unsigned int  numRows,
        unsigned int  numCols,
        int*          completedRows)
{
    // Use volatile so that updates from other blocks are always visible
    volatile int* rowReady_v = (volatile int*)rowReady;

    // shared variable used to broadcast the claimed row to all threads
    __shared__ int sharedRow;

    unsigned int nB = numCols;

    if (numRows == 0) return;

    // spread starting positions across blocks to reduce contention
    unsigned int step      = (numRows + gridDim.x - 1) / gridDim.x;
    unsigned int scanStart = (blockIdx.x * step) % numRows;

    while (true) {

        // only thread 0 searches for a row to process
        if (threadIdx.x == 0) {
            sharedRow = -1;

            unsigned int pos = scanStart;

            // scan for a ready row
            for (unsigned int checked = 0; checked < numRows; ++checked) {

                if (rowReady_v[pos] == 1) {

                    // try to claim it
                    int old = atomicCAS(&rowReady[pos], 1, 2);

                    if (old == 1) {
                        sharedRow  = (int)pos;
                        scanStart  = (pos + 1) % numRows;
                        break;
                    }
                }

                if (++pos >= numRows) pos = 0;
            }

            // if nothing found, check if all rows are done
            if (sharedRow == -1 &&
                atomicAdd(completedRows, 0) >= (int)numRows) {
                sharedRow = -2;
            }
        }

        // broadcast result to all threads
        __syncthreads();

        if (sharedRow == -2) break;     // everything done
        if (sharedRow == -1) continue;  // try again

        unsigned int i = (unsigned int)sharedRow;

        // compute X(i,b) in parallel over RHS columns
        for (unsigned int b = threadIdx.x; b < nB; b += blockDim.x) {

            float sum  = B->values[i * nB + b];
            float diag = 1.0f;

            // traverse row i
            for (unsigned int idx = L_r->rowPtrs[i];
                              idx < L_r->rowPtrs[i + 1]; ++idx) {

                unsigned int col = L_r->colIdxs[idx];
                float val = L_r->values[idx];

                // dependency already resolved
                if (col < i) {
                    sum -= val * X->values[col * nB + b];
                }
                // diagonal
                else if (col == i) {
                    diag = (val != 0.0f) ? val : 1.0f;
                }
            }

            X->values[i * nB + b] = sum / diag;
        }

        // make sure all threads finished writing X(i,*)
        __syncthreads();

        // thread 0 updates dependents
        if (threadIdx.x == 0) {

            // ensure writes to X are visible before releasing next rows
            __threadfence();

            for (unsigned int idx = dependentOffsets[i];
                              idx < dependentOffsets[i + 1]; ++idx) {

                unsigned int j = dependents[idx];

                // decrease remaining dependency count
                int rem = atomicSub(&depCounter[j], 1);

                // if this was the last dependency, row becomes ready
                if (rem == 1) {
                    atomicExch(&rowReady[j], 1);
                }
            }

            // mark row i as completed
            atomicAdd(completedRows, 1);
        }

        __syncthreads();
    }
}


void sptrsv_gpu0_v3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                    CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;

    if (n == 0 || numCols == 0) return;

    // count dependencies for each row
    int* depCount_h = (int*)calloc(n, sizeof(int));
    unsigned int* dependentCount_h = (unsigned int*)calloc(n, sizeof(unsigned int));

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i];
                          idx < L_r_host->rowPtrs[i + 1]; ++idx) {

            unsigned int col = L_r_host->colIdxs[idx];

            if (col < i) {
                depCount_h[i]++;
                dependentCount_h[col]++;
            }
        }
    }

    // build prefix sum for dependents
    unsigned int* dependentOffsets_h =
        (unsigned int*)malloc((n + 1) * sizeof(unsigned int));

    dependentOffsets_h[0] = 0;
    for (unsigned int j = 0; j < n; ++j) {
        dependentOffsets_h[j + 1] =
            dependentOffsets_h[j] + dependentCount_h[j];
    }

    unsigned int totalDeps = dependentOffsets_h[n];

    // build dependents list
    unsigned int* dependents_h =
        (unsigned int*)malloc((totalDeps > 0 ? totalDeps : 1) * sizeof(unsigned int));

    unsigned int* fillPos = (unsigned int*)calloc(n, sizeof(unsigned int));

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int idx = L_r_host->rowPtrs[i];
                          idx < L_r_host->rowPtrs[i + 1]; ++idx) {

            unsigned int col = L_r_host->colIdxs[idx];

            if (col < i) {
                unsigned int pos = dependentOffsets_h[col] + fillPos[col];
                dependents_h[pos] = i;
                fillPos[col]++;
            }
        }
    }

    free(fillPos);
    free(dependentCount_h);

    // initialize ready rows
    int* rowReady_h = (int*)calloc(n, sizeof(int));
    for (unsigned int i = 0; i < n; ++i) {
        if (depCount_h[i] == 0) rowReady_h[i] = 1;
    }

    // allocate GPU memory
    int *depCounter_d, *rowReady_d, *completedRows_d;
    unsigned int *dependents_d, *dependentOffsets_d;

    CUDA_ERROR_CHECK(cudaMalloc(&depCounter_d, n * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&rowReady_d, n * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&dependents_d,
                                (totalDeps > 0 ? totalDeps : 1) * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc(&dependentOffsets_d,
                                (n + 1) * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc(&completedRows_d, sizeof(int)));

    CUDA_ERROR_CHECK(cudaMemcpy(depCounter_d, depCount_h,
                                n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(rowReady_d, rowReady_h,
                                n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(dependents_d, dependents_h,
                                (totalDeps > 0 ? totalDeps : 1) * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(dependentOffsets_d, dependentOffsets_h,
                                (n + 1) * sizeof(unsigned int),
                                cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemset(completedRows_d, 0, sizeof(int)));

    // choose number of blocks based on number of SMs
    int numSMs = 1;
    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, 0));

    const unsigned int blockSize = 128;

    unsigned int gridSize = (unsigned int)(numSMs * 2);
    if (gridSize > n) gridSize = n;
    if (gridSize == 0) gridSize = 1;

    // launch kernel
    sptrsv_gpu0_kernel_v3<<<gridSize, blockSize>>>(
        L_r, B, X,
        depCounter_d,
        rowReady_d,
        dependents_d,
        dependentOffsets_d,
        n,
        numCols,
        completedRows_d
    );

    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // free everything
    CUDA_ERROR_CHECK(cudaFree(depCounter_d));
    CUDA_ERROR_CHECK(cudaFree(rowReady_d));
    CUDA_ERROR_CHECK(cudaFree(dependents_d));
    CUDA_ERROR_CHECK(cudaFree(dependentOffsets_d));
    CUDA_ERROR_CHECK(cudaFree(completedRows_d));

    free(depCount_h);
    free(rowReady_h);
    free(dependents_h);
    free(dependentOffsets_h);
}
