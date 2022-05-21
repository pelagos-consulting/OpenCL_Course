// Example of a global variable
#ifdef __opencl_c_program_scope_global_variables
__global int a_g = 2; 
__global float b_g[2] = {2.0,1.0}; 
#endif

// Example of constant memory
__constant float pi = 3.1415;
__constant float coeffs[] = {1.0, -2.0, 1.0};

// Kernel function to get the start and end values
// for filling a shared memory array
void get_start_end(
    size_t local_length, 
    size_t array_length,
    size_t local_index,
    size_t *start,
    size_t *end) {
  
    // Work out the jumps
    size_t jump=array_length/local_length;
    if (array_length%local_length) jump++;
    *start=local_index*jump;
    *end=(local_index+1)*jump;
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
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Scratch variable
    double temp=0.0; 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A[i0*N1_A+n]*B[n*N1_C+i1]; 
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
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Scratch variable
    float temp=0.0; 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A[i0*N1_A+n]*B[n*N1_C+i1]; 
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
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Scratch variable
    float temp=0.0;
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Implement prefetching for A
        __global float* A_i0 = &A[i0*N1_A];
        prefetch(A_i0, (size_t)N1_A);
    
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            //temp+=A[i0*N1_A+n]*B[n*N1_C+i1];
            temp += A_i0[n]*B[n*N1_C+i1];
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
} 

// matrix multiply kernel with pre-fetching
__kernel void mat_mult_transpose_A (__global float* AT, 
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
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Scratch variable
    float temp=0.0;
    
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

// matrix multiply kernel with pre-fetching
__kernel void mat_mult_transpose_B (__global float* A, 
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
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Scratch variable
    float temp=0.0;
    
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

// Matrix multiply kernel that uses local memory
__kernel void mat_mult_patch (
                        __global float* A_star, 
                        __global float* BT_star, 
                        __global float* C_star,
                        unsigned int N1_A_star, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A_star is of size (N0_C, N1_A_star), (i1,i2)
    // BT_star is of size (N1_C, N1_A_star), (i1, i2)
    // C_star is of size (N1_A_v, N0_C, N1_C), (i0, i1, i2)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i2=get_global_id(0); // Fastest dimension
    size_t i1=get_global_id(1); 
    size_t i0=get_global_id(2); // Slowest dimension
    
    // Get the number of vector elements
    size_t N1_A_v = get_global_size(2);
    
    // start and end along N1_A_star
    size_t start, end;
    
    // Get the start and end lengths to fill a block
    get_start_end(N1_A_v, N1_A_star, i0, &start, &end);
    
    // Scratch variable
    float temp=0.0;
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i1<N0_C) && (i2<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=start; n<end; n++) {
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A_star[i1*N1_A_star+n]*BT_star[i2*N1_A_star+n]; 
        } 
        // Number of rows in C is same as number of rows in A
        C_star[i0*N0_C*N1_C+i1*N1_C+i2]=temp;
    }
}

// Matrix multiply kernel that uses local memory
__kernel void mat_mult_patch_local (
                        __global float* A_star, 
                        __global float* BT_star, 
                        __global float* C_star,
                        __local float* shared_A_star,
                        __local float* shared_BT_star,
                        unsigned int N1_A_star, 
                        unsigned int N0_C,
                        unsigned int N1_C,
                        unsigned int vector_len) { 
    
    // A_star is of size (N0_C, N1_A_star), (i1, n)
    // BT_star is of size (N1_C, N1_A_star), (i2, n)
    // C_star is of size (N1_A_v, N0_C, N1_C), (i0, i1, i2)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i2=get_global_id(0); // Fastest dimension
    size_t i1=get_global_id(1); 
    size_t i0=get_global_id(2); // Slowest dimension
    
    // shared_A_star is of size (L0, vector_len) (s0, n)
    // shared_BT_star is of size (L1, vector_len) (s1, n)
    size_t L0 = get_local_size(1); // Slowest dimension
    size_t L1 = get_local_size(0); // Fastest dimension
    
    // index within local memory
    size_t s0 = get_local_id(1); // Slowest dimension
    size_t s1 = get_local_id(0); // fastest dimension
    
    // Get the number of vector elements
    size_t N1_A_v = get_global_size(2);
    
    // start and end along N1_A_star
    size_t start, end;
    
    // Get the start and end lengths of the subsection along N1_A_star
    get_start_end(N1_A_v, N1_A_star, i0, &start, &end);
    
    // Fetch local memory into shared_A_star and shared_BT_star
    
    // Fill the rows of shared_A_star and shared_BT_star
    // From row i1 of A_star
    if (i1<N0_C) {
        for (int n = start; n<end; n++) {
            shared_A_star[s0*vector_len+n-start] = A_star[i1*N1_A_star+n];
            //shared_A_star[s0*vector_len+n-start] = 2.0;
        }
    }
    // From row i2 of BT_star
    if (i2<N1_C) {
        for (int n = start; n<end; n++) {
            shared_BT_star[s1*vector_len+n-start] = BT_star[i2*N1_A_star+n];
            //shared_BT_star[s1*vector_len+n-start] = 2.0;
        }
    }       
    
    // Enqueue a local barrier to ensure shared memory is filled
    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Scratch variable
    float temp=0.0;
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i1<N0_C) && (i2<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<(end-start); n++) {
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=shared_A_star[s0*vector_len+n]*shared_BT_star[s1*vector_len+n];
            
        }
        
        // Number of rows in C is same as number of rows in A
        C_star[i0*N0_C*N1_C+i1*N1_C+i2]=temp;
    }
}

__kernel void c_star_stack (
                        __global float* C_star,
                        __global float* C,
                        unsigned int N1_A_v, 
                        unsigned int N0_C,
                        unsigned int N1_C) {    

    // C_star is of size (N1_A_v, N0_C, N1_C) (n, i0, i1)
    // C is of size (N0_C, N1_C) (i0, i1)
    size_t i0=get_global_id(1); // Slowest dimension
    size_t i1=get_global_id(0); // Fastest dimension
    
    // Temporary storage
    float temp=0.0;
    
    if ((i0<N0_C) && (i1<N1_C)) {    
        for (int n=0; n<N1_A_v; n++) {
            temp+=C_star[n*N0_C*N1_C+i0*N1_C+i1];
        }
        C[i0*N1_C+i1]=temp;
    }
}

// Matrix multiply kernel that uses local memory
__kernel void mat_mult_local (
                        __global float* A, 
                        __global float* B, 
                        __global float* C,
                        __local  float* shared_A,
                        __local  float* shared_B,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)
    
    // Make a local scratch array for demonstration purposes
    // (not actually used)
    __local float scratch[10];
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Location within the workgroup
    size_t s0=get_local_id(0);
    size_t s1=get_local_id(1);
    
    // Local size
    size_t L0=get_local_size(0);
    size_t L1=get_local_size(1);
    
    // start and end
    size_t start0, end0, start1, end1;
    
    // Fill shared memory
    
    // Get the start1 and end1 lengths to fill a block
    get_start_end(L1, N1_A, s1, &start1, &end1);
    // Fill shared_A with the rows of A
    if (i0<N0_C) {
        for (size_t n=start1; n<end1; n++) {
            shared_A[s0*N1_A+n]=A[i0*N1_A+n]; 
        }
    }   
    
    // Get the start0 and end0 lengths
    get_start_end(L0, N1_A, s0, &start0, &end0);
    // Fill the columns of shared with B
    if (i1<N1_C) {
        for (size_t n=start0; n<end0; n++) {
            shared_B[s1*N1_A+n]=B[n*N1_C+i1]; 
        }
    }
    
    // Enqueue a local barrier to make sure shared memory is filled
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Scratch variable whose allocation uses constant memory pi
    float temp=0.0*pi; 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=shared_A[s0*N1_A+n]*shared_B[s1*N1_A+n]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}

// Local memory matrix multiply kernel
// where B has been transposed
__kernel void mat_mult_local_transp (
                        __global float* A, 
                        __global float* BT, 
                        __global float* C,
                        __local  float* shared_A,
                        __local  float* shared_BT,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A is of size (N0_C, N1_A)
    // BT is of size (N1_C, N1_A)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Location within the workgroup
    size_t s0=get_local_id(0);
    size_t s1=get_local_id(1);
    
    // Local size
    size_t L0=get_local_size(0);
    size_t L1=get_local_size(1);
    
    // start and end
    size_t start0, end0, start1, end1;
    
    // Fill shared memory
    
    // Get the start1 and end1 lengths to fill a block
    get_start_end(L1, N1_A, s1, &start1, &end1);
    // Fill shared_A with the rows of A
    if (i0<N0_C) {
        for (size_t n=start1; n<end1; n++) {
            shared_A[s0*N1_A+n]=A[i0*N1_A+n]; 
        }
    }   
    
    // Get the start0 and end0 lengths
    get_start_end(L0, N1_A, s0, &start0, &end0);
    // Fill the columns of shared with B
    if (i1<N1_C) {
        for (size_t n=start0; n<end0; n++) {
            shared_BT[s1*N1_A+n]=BT[i1*N1_A+n]; 
        }
    }
    
    // Enqueue a local barrier to make sure shared memory is filled
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Scratch variable whose allocation uses constant memory pi
    float temp=0.0; 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // BT is of size (N1_C, N1_A)
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=shared_A[s0*N1_A+n]*shared_BT[s1*N1_A+n]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}

// Local memory matrix multiply kernel
// where B has been transposed and using vectors
__kernel void mat_mult_local_transp_vec (
                        __global float8* A_star, 
                        __global float8* BT_star, 
                        __global float* C,
                        __local  float8* shared_A_star,
                        __local  float8* shared_BT_star,
                        unsigned int N1_A_v, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A_star is of size (N0_C, N1_A_v)
    // BT_star is of size (N1_C, N1_A_v)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1);
    
    //printf("%zu %zu\n", i0, i1);
    
    // Location within the workgroup
    size_t s0=get_local_id(0);
    size_t s1=get_local_id(1);
    
    // Size of the workgroup
    size_t L0=get_local_size(0);
    size_t L1=get_local_size(1);
    
    // start and end
    size_t start0, end0, start1, end1;
    
    // Get the start1 and end1 lengths to fill a block
    get_start_end(L1, N1_A_v, s1, &start1, &end1);
    // Fill shared_A with the rows of A
    if (i0<N0_C) {
        for (size_t n=start1; n<end1; n++) {
            shared_A_star[s0*N1_A_v+n]=A_star[i0*N1_A_v+n]; 
        }
    }   
    
    // Get the start0 and end0 lengths
    get_start_end(L0, N1_A_v, s0, &start0, &end0);
    // Fill the rows of shared_BT_star with BT_star
    if (i1<N1_C) {
        for (size_t n=start0; n<end0; n++) {
            shared_BT_star[s1*N1_A_v+n]=BT_star[i1*N1_A_v+n]; 
        }
    }
    
    // Enqueue a local barrier to make sure shared memory is filled
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Scratch variable whose allocation uses constant memory pi
    float8 temp=(float8)(0.0f); 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of shared_A_star 
        // and columns of BT_star 
        for (size_t n=0; n<N1_A_v; n++) {
            
            // Local size
            // shared_A_star is of size (L0, N1_A_v)
            // shared_BT_star is of size (L1, N1_A_v)    
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=shared_A_star[s0*N1_A_v+n]*shared_BT_star[s1*N1_A_v+n]; 
        } 
        
        // Number of rows in C is same as number of rows in A
        
        // sum over the elements of the vector
        C[i0*N1_C+i1]=(
            temp.s0+temp.s1+temp.s2+temp.s3
            +temp.s4+temp.s5+temp.s6+temp.s7
        );
    }
}


// Local memory matrix multiply kernel 
__kernel void transpose (__global float* src, 
                        __global float* dest, 
                        unsigned int N0_src,
                        unsigned int N1_src) { 
    
    // src is of size (N0_src, N1_src)
    // dest is of size (N1_src, N0_src)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    if ((i0<N0_src) && (i1<N1_src)) {
        dest[i1*N0_src+i0]=src[i0*N1_src+i1];
    }
}
