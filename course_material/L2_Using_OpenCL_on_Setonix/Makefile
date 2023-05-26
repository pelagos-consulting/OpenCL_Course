
# Include general environment variables
include ../env

# List of applications to target
TARGETS=hello_devices_mpi.exe hello_jobstep.exe

all: $(TARGETS)

# Compile the kernel source only with hipcc
%.exe : %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS) 

# Clean step
clean:
	rm -rf *.exe
	rm -rf *.o


.EXPORT_ALL_VARIABLES:
