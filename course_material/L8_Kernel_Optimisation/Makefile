# Set DEBUG=true to enable debugging
# DEBUG=true

# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult_float.exe \
		mat_mult_double.exe \
		mat_mult_prefetch.exe \
		mat_mult_local.exe \
		mat_mult_AT.exe \
		mat_mult_BT.exe \
		mat_mult_tile_AT.exe \
	   	mat_mult_tile_BT.exe \
		mat_mult_tile_vector_BT.exe \
		mat_mult_tile_local_BT.exe \
		mat_mult_tile_local_vector_BT.exe \
		mat_mult_clblast.exe \
		mat_mult_clblast_md.exe \
		mat_mult_local_A.exe         

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Specific compilation step for CLBlast codes
mat_mult_clblast.exe: mat_mult_clblast.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS) -lclblast

mat_mult_clblast_md.exe: mat_mult_clblast_md.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS) -lclblast

# Clean step
clean:
	rm -rf *.exe


.EXPORT_ALL_VARIABLES:
