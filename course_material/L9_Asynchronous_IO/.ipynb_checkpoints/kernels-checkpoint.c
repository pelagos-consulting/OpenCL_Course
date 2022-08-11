// Kernel to solve the wave equation with fourth-order accuracy
__kernel void wave2d_4o (
        __global float* U0,
        __global float* U1,
        __global float* U2,
        __global float* V,
        unsigned int N0,
        unsigned int N1,
        float dt2,
        float inv_dx02,
        float inv_dx12) {    

    // C_star is of size (N1_A_c, N0_C, N1_C) (n, i0, i1)
    // C is of size (N0_C, N1_C) (i0, i1)
    size_t i0=get_global_id(1); // Slowest dimension
    size_t i1=get_global_id(0); // Fastest dimension
    
    // Required padding and numbers of coefficients
    const int pad_l=2, pad_r=2, ncoeffs=5;
    
    // Limits
    i0=min(i0, (size_t)(N0-1-pad_r));
    i1=min(i1, (size_t)(N1-1-pad_r));
    i0=max((size_t)pad_l, i0);
    i1=max((size_t)pad_l, i1);
    
    // Finite difference coefficients for space derivative
    float coeffs[ncoeffs] = {-0.083333336f, 1.3333334f, -2.5f, 1.3333334f, -0.083333336f};
    
    // Temporary storage for the finite difference coefficient
    float temp0=0.0f, temp1=0.0f, tempV=V[offset];
    float tempU0 = U0[offset];
    float tempU1 = U1[offset];
    
    // Position within the array
    long offset=i0*N1+i1;
    
    // Calculate the Laplacian
    #pragma unroll
    for (int n=0; n<ncoeffs; n++) {
        // Stride in dim0 is N1
        temp0+=coeffs[n]*U1[offset+(n-pad_l)*N1];
        // Stride in dim1 is 1
        temp1+=coeffs[n]*U1[offset+n-pad_l];
    }
    
    // Update the solution
    U2[offset]=2.0f*tempU1-tempU0
        -(dt2*tempV*tempV)*(temp0*inv_dx02+temp1*invdx12);
}