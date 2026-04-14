// kernel.cu


#include "common.h"


// L: sparse lower triangular matrix stored in CSR format
// B: dense right-hand side matrix
// X: dense output solution matrix
// The system being solved is: L * X = B
// CSR is more efficient for a sparse matrix since it only stores non-zero values and their corresponding row and column indices, which reduces memory usage and can speed up computations by skipping zero entries
void sptrsv_cpu(CSRMatrix* L, DenseMatrix* B, DenseMatrix* X){

    // n = number of rows in L
    // Since L is the coefficient matrix, this is also the number of equations
    unsigned int n = L->numRows;
    // nB = number of columns in B
    // Each column of B represents one right-hand side,
    unsigned int nB = B->numCols;

    // Loop over each right-hand side (each column of B)
    // For each b, we solve:
    //      L * x^(b) = b^(b)
    for(unsigned int b = 0; b < nB; ++b){
        // Forward substitution loop for the b-th right-hand side (b-th system); moving one equation at a time in the forward direction
        // Forward substitution over the rows of L
        // Since L is lower triangular, row i depends only on previously computed rows 0, 1, ..., i-1
        for(unsigned int i = 0; i < n; ++i){
            // Start with the current entry of the right-hand side: sum = B(i, b)
            // This value will later be reduced by subtracting the known contributions from previously computed solution entries
            float sum = B->values[i * nB + b];
            // This variable will store the diagonal entry L(i,i) (the coefficient of the target variable i particular iteration), which is needed at the end to divide the sum and solve for X(i,b)
            float diag = 0.0f;
            
            // Traverse all nonzero entries in row i of L
            for(unsigned int idx = L->rowPtrs[i]; idx < L->rowPtrs[i + 1]; ++idx){
                // Column index of the current nonzero entry
                unsigned int col = L->colIdxs[idx];
                // Value of the current nonzero entry
                float val = L->values[idx];

                // If col < i, this entry is below the diagonal (before the target variable in a certain iteration)
                // It corresponds to a known term L(i,col) * X(col,b) from a previously solved variable, so we simply subtract it from sum (equivalent to moving the term from the left-hand side to the right-hand side): sum = sum - L(i,col) * X(col,b)
                if(col < i){
                    sum -= val * X->values[col * nB + b];
                // If col == i, this is the diagonal entry L(i,i)
                // We store it in "diag" so that after subtracting all previous contributions, we can divide the sum by the diagonal and solve for X(i,b)
                } else if(col == i){
                    diag = val !=0 ? val : 1.0f; // Avoid division by zero
                }
            }
            // After processing the row, we now have: sum = B(i,b) - sum of known lower-triangular contributions
            // So we solve for the current unknown using: X(i,b) = sum / L(i,i)
            X->values[i * nB + b] = sum / diag;
        }
    }
}