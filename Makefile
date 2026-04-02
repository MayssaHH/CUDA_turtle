
NVCC        = nvcc
NVCC_FLAGS  = -O3
# Temporary: real GPU objects are kernel0.o … kernel3.o when those .cu files exist.
OBJ         = main.o matrix.o kernelCPU0.o kernel_stubs.o
EXE         = sptrsv


default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

