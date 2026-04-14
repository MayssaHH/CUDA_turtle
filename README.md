# Sparse Triangular Solve on CUDA (SpTRSV)

This project explores how far we can push a sparse triangular solve (SpTRSV) on modern GPUs by exploiting sparsity and parallelism.  The core problem is solving `L * X = B` where `L` is a sparse lower‑triangular matrix and `B` is a dense block of right‑hand sides.  By storing only the non‑zeros (CSC/CSR) we cut memory traffic dramatically; the challenge is recovering parallelism despite row dependencies.

## Implementations in this repo
- **CPU baseline** (`sptrsv_cpu` in `kernelCPU0.cu`): straightforward forward substitution over CSR.
- **GPU v1** (`sptrsv_gpu0_kernel_v1` / `sptrsv_gpu0_v1` in `kernel0_v1.cu`): one thread per RHS column; rows are still solved sequentially inside each thread. Easiest port, limited parallelism.
- **GPU v2** (`sptrsv_gpu0_kernel_v2` / `sptrsv_gpu0_v2` in `kernel0_v2.cu`): level‑set approach. The host builds dependency levels; rows in the same level run in parallel (2D grid: rows×columns) with global sync between levels.
- **GPU v3** (`sptrsv_gpu0_kernel_v3` / `sptrsv_gpu0_v3` in `kernel0_v3.cu`): dynamic dependency scheduler. Rows become ready as their prerequisites finish; blocks claim ready rows and update dependents—reducing global barriers and increasing concurrency.

## Datasets
Place the SuiteSparse test matrices (or equivalents) under `data/`:
- `rajat18.txt` — small.
- `parabolic_fem.txt` — medium.
- `tmt_sym.txt` — large.

Each file is expected in the simple text format read by `createCSCMatrixFromFile`: first line `numRows numNonzeros`, followed by `row col value` per non‑zero (0‑based indices). Missing diagonals are auto‑inserted.

## Repo layout (top level)
- `main.cu` — argument parsing, dataset loading, run/verify CPU & GPU variants, timing.
- `matrix.cu` / `matrix.h` — sparse/dense data structures, file I/O, host↔device transfers.
- `kernelCPU0.cu` — CPU solver.
- `kernel0_v1.cu`, `kernel0_v2.cu`, `kernel0_v3.cu` — GPU variants.
- `common.h`, `timer.h`, `Makefile`.

## Building
Requirements: CUDA toolkit (`nvcc`) and a POSIX‑style build environment (WSL/MinGW on Windows is fine).

```bash
git clone https://github.com/MayssaHH/CUDA_turtle.git
cd CUDA_turtle
make           # produces ./sptrsv
```

If `getopt`/`unistd` headers are missing on Windows, build under WSL or replace with Windows equivalents.

## Running
Flags:
- `-d {s|m|l}` choose dataset (`s`=rajat18, `m`=parabolic_fem, `l`=tmt_sym).
- `-s` run CPU baseline.
- `-0`, `-1`, `-2` enable GPU v1, v2, v3 respectively (multiple can be combined).

Examples:
```bash
./sptrsv -d s -s                 # CPU only, small dataset
./sptrsv -d m -s -0 -1 -2        # CPU + all GPU variants on medium
./sptrsv -d l -0                 # GPU v1 only on large
```

Outputs include per‑kernel timings and verification against CPU results.

## CPU Runtime (ms)

| Data Set Size | Small  | Medium  | Large   |
|--------------|--------|---------|---------|
| 128 cols     | 294.58 | 2762.80 | 2354.23 |
| 256 cols     | 609.28 | 5719.94 | 5147.95 |
| 512 cols     | 1543.28| 11774.07| 13524.27 |

---

## Kernel0 Version 1 Runtime (ms)

| Data Set Size | Small | Medium | Large |
|--------------|-------|--------|-------|
| 128 cols     | 88.39 | 535.05 | 565.91 |
| 256 cols     | 85.80 | 541.72 | 578.08 |
| 512 cols     | 84.78 | 534.45 | 568.86 |

---

## Kernel0 Version 2 Runtime (ms)

| Data Set Size | Small | Medium | Large |
|--------------|-------|--------|-------|
| 128 cols     | 6.91  | 16.44  | 5233.08 |
| 256 cols     | 7.40  | 21.14  | 5321.99 |
| 512 cols     | 8.98  | 32.97  | 5505.31 |

---

## Kernel0 Version 3 Runtime (ms)

| Data Set Size | Small | Medium | Large |
|--------------|-------|--------|-------|
| 128 cols     | 48.71 | 224.20 | 1922.77 |
| 256 cols     | 51.83 | 219.76 | 2355.89 |
| 512 cols     | 56.71 | 233.28 | 3263.96 |
## Reproducing the table
1) Build as above.  
2) Ensure `data/` contains the three matrices named above.  
3) Run: `./sptrsv -d m -s -0 -1 -2` (replace `m` with `s` or `l` as desired).  
4) Collect the printed timings and verification messages.

## Attribution
Created for exploring sparse linear algebra on GPUs; dataset examples come from the SuiteSparse Matrix Collection.
