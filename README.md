# Sparse Triangular Solve on CUDA (SpTRSV)

This project explores sparse triangular solve (SpTRSV) on CUDA for systems of the form `L * X = B`, where `L` is a sparse lower-triangular matrix and `B` is a dense block of right-hand sides.

## Implementations
- **CPU baseline** (`sptrsv_cpu` in `kernelCPU0.cu`): straightforward forward substitution over CSR.
- **GPU variant 0** (`sptrsv_gpu0` in `kernel0.cu`): host-computed level-set scheduling with one kernel launch per level.
- **GPU variant 1** (`sptrsv_gpu1` in `kernel1.cu`): level-set scheduling plus shared-memory tiling of sparse rows.
- **GPU variant 2** (`sptrsv_gpu2` in `kernel2.cu`): tiled level-set solver launched on a CUDA stream.
- **GPU variant 3** (`sptrsv_gpu3` in `kernel3.cu`): hybrid experimental path that switches between a level-set kernel and a thin one-thread-per-column kernel.

## Datasets
Place the input matrices under `data/`:
- `rajat18.txt`
- `parabolic_fem.txt`
- `tmt_sym.txt`

Each file is read by `createCSCMatrixFromFile` in this format:
- first line: `numRows numNonzeros`
- remaining lines: `row col value`

Indices are 0-based. Missing diagonal entries are inserted automatically.

## Repo Layout
- `main.cu`: argument parsing, dataset loading, CPU/GPU execution, timing, verification.
- `matrix.cu` / `matrix.h`: sparse and dense matrix structures, file I/O, host/device allocation, transfers.
- `kernelCPU0.cu`: CPU solver.
- `kernel0.cu`, `kernel1.cu`, `kernel2.cu`, `kernel3.cu`: GPU solver variants.
- `common.h`, `timer.h`, `Makefile`: shared declarations, timing helper, build script.

## Build
Requirements:
- CUDA toolkit with `nvcc`
- a POSIX-style environment for `make`

```bash
make
```

This produces `sptrsv`.

## Run
Flags:
- `-d {s|m|l}` selects the dataset (`s` = `rajat18`, `m` = `parabolic_fem`, `l` = `tmt_sym`)
- `-s` runs the CPU baseline
- `-0`, `-1`, `-2`, `-3` enable GPU variants 0, 1, 2, 3

Examples:

```bash
./sptrsv -d s -s
./sptrsv -d m -s -0 -1 -2 -3
./sptrsv -d l -0
```

The program prints timings and verifies GPU results against the CPU baseline.
