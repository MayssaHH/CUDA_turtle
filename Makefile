
NVCC       = nvcc
NVCC_FLAGS = -O3

EXE        = sptrsv
SOURCES    = main.cu matrix.cu kernelCPU0.cu kernel0_v1.cu kernel0_v2.cu kernel0_v3.cu
OBJ        = $(SOURCES:.cu=.o)

default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $@

.PHONY: clean
clean:
	rm -rf $(OBJ) $(EXE)
