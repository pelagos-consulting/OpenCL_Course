    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_elementwise", &errcode);
    H_ERRCHK(errcode);
    
    // Set arguments to the kernel (not thread safe)
    H_ERRCHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &D_d));
    H_ERRCHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &E_d));
    H_ERRCHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_d));
    H_ERRCHK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &N0_F));
    H_ERRCHK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &N1_F));
