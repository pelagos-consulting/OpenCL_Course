// Example of a global variable
#ifdef __opencl_c_program_scope_global_variables
__global int a_g = 2; 
__global float b_g[2] = {2.0f,1.0f}; 
#endif

// Example of constant memory
__constant float pi = 3.1415f;
__constant float coeffs[] = {1.0f, -2.0f, 1.0f};

// Kernel function to get the start and end values
// for filling a shared memory array
void get_start_end(
    // Number of work-items along a dimension of workgroup
    size_t local_length,
    // Number of items in the array
    size_t array_length,
    // Index of work item along dimension of workgroup
    size_t local_index,
    // Starting position of the copy
    size_t *start,
    // End position of the copy
    size_t *end) {
  
    // Work out the jump size
    size_t jump_size=array_length/local_length;
    if (array_length%local_length) jump_size++;
    
    // Starting position for the copy
    *start=local_index*jump_size;
    // End position for the copy
    *end=(local_index+1)*jump_size;
    // Limit end so we don't go off the end
    *end=min(*end,array_length);
}    

// standard matrix multiply kernel 
__kernel void mat_mult_double (
                        __global double* A, 
                        __global double* B, 
                        __global double* C, 
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Scratch variable
    double temp=0.0;

    __global double* A_i0 = &A[i0*N1_A];
    __global double* B_i1 = &B[i1];

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A_i0[n]*B_i1[n*N1_C]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
} 

// standard matrix multiply kernel 
__kernel void mat_mult_float (__global float* A, 
                        __global float* B, 
                        __global float* C, 
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Scratch variable
    float temp=0.0f; 

    __global float* A_i0 = &A[i0*N1_A];
    __global float* B_i1 = &B[i1];
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A_i0[n]*B_i1[n*N1_C]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
} 

// matrix multiply kernel with pre-fetching
__kernel void mat_mult_prefetch (__global float* A, 
                        __global float* B, 
                        __global float* C, 
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Scratch variable
    float temp=0.0f;
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Implement prefetching for A
        __global float* A_i0 = &A[i0*N1_A];
        __global float* B_i1 = &B[i1];
        prefetch(A_i0, (size_t)N1_A);
    
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            //temp+=A[i0*N1_A+n]*B[n*N1_C+i1];
            temp += A_i0[n]*B_i1[n*N1_C];
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}

// Matrix multiply kernel that uses local memory for B
__kernel void mat_mult_local_B (
                        __global float* A, 
                        __global float* B, 
                        __global float* C,
                        __local  float* shared_B,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // shared_B is of size (L1, N1_A)
    // C is of size (N0_C, N1_C)
    
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Location within the workgroup
    size_t s0=get_local_id(1);
    size_t s1=get_local_id(0);
    
    // Local size
    size_t L0=get_local_size(1);
    size_t L1=get_local_size(0);
    
    // start and end
    size_t start, end;
    
    // Fill shared_B
    
    // Get the start and end lengths
    get_start_end(L0, N1_A, s0, &start, &end);
    // Fill the columns of shared with B
    if (i1<N1_C) {
        for (size_t n=start; n<end; n++) {
            shared_B[s1*N1_A+n]=B[i1+n*N1_C]; 
        }
    }
    
    // Enqueue a local barrier to make sure all the work items finish
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Scratch variable
    float temp=0.0f; 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            // shared_B is of size (L1, N1_A)
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and across row s1 of shared_B
            temp+=A[i0*N1_A+n]*shared_B[s1*N1_A+n]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}

// Matrix multiply kernel that uses local memory for A
__kernel void mat_mult_local_A (
                        __global float* A, 
                        __global float* B, 
                        __global float* C,
                        __local  float* shared_A,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // shared_B is of size (L1, N1_A)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Location within the workgroup
    size_t s0=get_local_id(1);
    size_t s1=get_local_id(0);
    
    // Local size
    size_t L0=get_local_size(1);
    size_t L1=get_local_size(0);
    
    // start and end positions for the copy
    size_t start, end;
    
    // Get the start and end lengths to fill array
    get_start_end(L1, N1_A, s1, &start, &end);
    
    // Fill shared_A
    if (i0<N0_C) {
        for (size_t n=start; n<end; n++) {
            shared_A[s0*N1_A+n]=A[i0*N1_A+n]; 
        }   
    }
    
    // Enqueue a local barrier to make sure all work items reach here
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Scratch variable
    float temp=0.0f; 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            // shared_B is of size (L1, N1_A)
            // C is of size (N0_C, N1_C)
            
            // Loop across row s0 of shared_A
            // and down column i1 of B
            temp+=shared_A[s0*N1_A+n]*B[n*N1_C+i1];
            
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}

// matrix multiply kernel with pre-fetching
__kernel void mat_mult_BT (__global float* A, 
                        __global float* BT, 
                        __global float* C, 
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // C is of size (N0_C, N1_C)
    // A is of size (N0_C, N1_A)
    // BT is of size (N1_C, N1_A)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Scratch variable
    float temp=0.0f;
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Implement prefetching for A
        __global float* A_i0 = &A[i0*N1_A];
        //prefetch(A_i0, (size_t)N1_A);

        __global float* BT_i1 = &BT[i1*N1_A];
        //prefetch(B_i1, (size_t)N1_A);
    
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and across row i1 of B
            temp += A_i0[n]*BT_i1[n];
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
} 

// matrix multiply kernel with pre-fetching
__kernel void mat_mult_AT (__global float* AT, 
                        __global float* B, 
                        __global float* C, 
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // C is of size (N0_C, N1_C)
    // AT is of size (N1_A, N0_C)
    // B is of size (N1_A, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Scratch variable
    float temp=0.0f;
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Implement prefetching for A
        __global float* AT_i0 = &AT[i0];
        //prefetch(A_i0, (size_t)N1_A);

        __global float* B_i1 = &B[i1];
        //prefetch(B_i1, (size_t)N1_A);
    
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and across row i1 of B
            temp += AT_i0[n*N0_C]*B_i1[n*N1_C];
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
} 


// Local memory matrix multiply kernel 
__kernel void transpose (__global float* src, 
                        __global float* dest, 
                        unsigned int N0_src,
                        unsigned int N1_src) { 
    
    // src is of size (N0_src, N1_src)
    // dest is of size (N1_src, N0_src)
    
    // i0 and i1 represent the coordinates in src 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    if ((i0<N0_src) && (i1<N1_src)) {
        dest[i1*N0_src+i0]=src[i0*N1_src+i1];
    }
}
