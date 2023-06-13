
# List of applications to target
TARGETS=mat_mult_profiling_mpi.exe

all: $(TARGETS)

# Compile the sources
%.exe : %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -rf *.exe

.EXPORT_ALL_VARIABLES:
