# Set DEBUG=true to enable debugging
# DEBUG=true

# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult_float.exe \
		mat_mult_double.exe \
		mat_mult_prefetch.exe \
		mat_mult_local_A.exe \
		mat_mult_local_B.exe \
		mat_mult_AT.exe \
		mat_mult_BT.exe \
		mat_mult_tile_local_AB.exe \
		mat_mult_tile_local_AB_vector.exe \
		mat_mult_tile_local_A.exe \
		mat_mult_tile_local_A_vector.exe \
		mat_mult_tile_local_B.exe \
		mat_mult_tile_local_B_vector.exe \
		mat_mult_clblast.exe \
		mat_mult_clblast_md.exe          

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
