# CUDA_turtle

## What this project is

CUDA_turtle is a **CUDA/C++ benchmark for the sparse lower-triangular solve** (SPTRSV). It solves **L X = B** where **L** is a sparse lower-triangular matrix and **B** is a dense matrix of right-hand sides. The program loads **L** from a text file, builds **CSC** and **CSR** views, generates random dense **B** with **128, 256, and 512 columns**, and times:

- a **CPU** forward-substitution reference (`sptrsv_cpu`), and later  
- up to **four GPU implementations** (`sptrsv_gpu0` … `sptrsv_gpu3`).

GPU results are checked against the CPU solution.

## Project structure

| Path | Role |
|------|------|
| `main.cu` | CLI, dataset selection, allocation, CPU/GPU orchestration, timing, verification |
| `common.h` | CUDA error macro, declarations for `sptrsv_cpu` and `sptrsv_gpu0`–`3` |
| `matrix.h` / `matrix.cu` | CSC/CSR/dense types, file I/O, host↔device copies, random dense **B** |
| `kernelCPU0.cu` | Sequential SPTRSV on CSR **L** (reference) |
| `kernel_stubs.cu` | No-op GPU entry points (placeholder until real kernels are added) |
| `timer.h` | Wall-clock timing helpers (`gettimeofday`) |
| `Makefile` | Builds the `sptrsv` executable with `nvcc` |
| `data/` | Matrix files in the format below |

**Matrix file format** (`data/*.txt`):

1. First line: `numRows numNonzeros` (unsigned integers).  
2. Each following line: `row col value` (row/column indices and coefficient; the loader also **adds any missing diagonal** with value `1.0` and sorts into CSC order).

Shipped examples:

- `data/tiny_ltri.txt` — small test matrix  
- `data/hard_ltri.txt` — default benchmark (`256` rows, `32896` nonzeros in the header)

The CLI also supports `-d s`, `-d m`, `-d l` for other named datasets (`rajat18`, `parabolic_fem`, `tmt_sym`); those files are **not** included here and must be placed under `data/` if you use those flags.

## Requirements

- **NVIDIA CUDA Toolkit** (`nvcc` on `PATH`)
- A **CUDA-capable GPU** only if you enable GPU flags (`-0` … `-3`); with stubs, GPU runs complete quickly but do **not** compute a real solve

## How to run

Clone the repository and enter the project directory:

```bash
git clone https://github.com/MayssaHH/CUDA_turtle.git
cd CUDA_turtle
```

Build the program (requires `nvcc` from the CUDA Toolkit):

```bash
make
```

That produces the **`sptrsv`** executable in the same directory. Run it:

```bash
./sptrsv
```

To remove object files and the binary, then rebuild:

```bash
make clean
make
```
