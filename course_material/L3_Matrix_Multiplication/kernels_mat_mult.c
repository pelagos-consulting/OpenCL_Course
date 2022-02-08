// standard matrix multiply kernel 
__kernel void mat_mult (__global float* A, 
                        __global float* B, 
                        __global float* C, 
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    float temp=0.0; 

    // Guard mechanism
    if ((i0<N0_C) && (i1<N1_C)) {
        // Loop over columns of A and rows of B 
        for (unsigned int n=0; n<N1_A; n++) { 
            // C has the same number of rows as A, 
            // and the same number of columns as B 
            // i0 is the row index of A 
            // i1 is the column index of B 
            temp+=A[i0*N1_A+n]*B[n*N1_C+i1]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp; 
    }
} 