# Sample makefile

#!include <win32.mak>

# Include general environment variables
!include ..\env.mak

# Location of general helper files
INC_DIR=..\include

# List of applications to target
TARGETS=mat_mult_float.exe mat_mult_double.exe mat_mult_local.exe mat_mult_prefetch.exe mat_mult_transpose_A.exe mat_mult_transpose_B.exe mat_mult_tile_local.exe mat_mult_tile_local_vector.exe mat_mult_chunk_vector.exe

all: $(TARGETS)

# General compilation step
.cpp.exe:
	$(CXX) $(CXXFLAGS) /I $(INC_DIR) $< $(OPENCL_LIB_FLAGS) -o $@  

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
