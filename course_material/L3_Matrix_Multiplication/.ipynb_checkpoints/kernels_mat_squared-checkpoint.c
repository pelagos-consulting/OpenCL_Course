// standard matrix multiply kernel 
__kernel void mat_squared (
                        __global float* C, 
                        __global float* D, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Create an offset
        size_t offset = i0*N1_C+i1;
        
        // Number of rows in C is same as number of rows in A
        D[offset]=C[offset]*C[offset];
    }
} 