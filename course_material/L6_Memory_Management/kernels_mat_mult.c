// Example of a global variable
#ifdef __opencl_c_program_scope_global_variables
__global int a_g = 2; 
__global float b_g[2] = {2.0,1.0}; 
#endif

// Example of constant memory
__constant float pi = 3.1415;
__constant float coeffs[] = {1.0, -2.0, 1.0};

// standard matrix multiply kernel 
__kernel void mat_mult (__global float* A, 
                        __global float* B, 
                        __global float* C,
                        __local  float* shared,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // C is of size (N0_C, N1_C)
    // B is of size (N1_A, N1_C)
    // A is of size (N0_C, N1_A)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Location in local (shared) memory
    size_t s0=get_local_id(0);
    size_t s1=get_local_id(1);
    
    // Local size
    size_t L0=get_local_size(0);
    
    // Work out the jumps
    size_t jump=N1_A/L0;
    if (N1_A%L0) jump++;
    size_t start = s0*jump;
    size_t end = (s0+1)*jump;
    end = min(end,(size_t)N1_A);
    
    // Fill the columns of shared with B
    if (i1<N1_C) {
        for (size_t n=start; n<end; n++) {
            shared[s1*N1_A+n] = B[n*N1_C+i1]; 
        }
    }
    
    // Scratch variable whose allocation uses constant memory
    float temp=0.0*pi; 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A[i0*N1_A+n]*shared[s1*N1_A+n]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}
