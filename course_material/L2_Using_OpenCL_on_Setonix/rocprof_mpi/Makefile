# List of applications to target
TARGETS=mat_mult_profiling_mpi.exe

all: $(TARGETS)

# Compile the sources
%.exe : %.cpp
	CC -g -O2 -fopenmp -I../../include $< -o $@ -lOpenCL

# Clean step
clean:
	rm -rf *.exe

.EXPORT_ALL_VARIABLES:
