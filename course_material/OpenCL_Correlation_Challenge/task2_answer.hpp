        //// Begin Task 2 - Code to set kernel arguments for each thread /////////
        
        // Set kernel arguments for kernels[n]
        H_ERRCHK(clSetKernelArg(kernels[n], 0, sizeof(cl_mem), &srcs_d[n]));
        H_ERRCHK(clSetKernelArg(kernels[n], 1, sizeof(cl_mem), &dsts_d[n]));
        H_ERRCHK(clSetKernelArg(kernels[n], 2, sizeof(cl_mem), &kerns_d[n]));
        H_ERRCHK(clSetKernelArg(kernels[n], 3, sizeof(cl_uint), &len0_src));
        H_ERRCHK(clSetKernelArg(kernels[n], 4, sizeof(cl_uint), &len1_src));
        H_ERRCHK(clSetKernelArg(kernels[n], 5, sizeof(cl_uint), &pad0_l));
        H_ERRCHK(clSetKernelArg(kernels[n], 6, sizeof(cl_uint), &pad0_r));
        H_ERRCHK(clSetKernelArg(kernels[n], 7, sizeof(cl_uint), &pad1_l));
        H_ERRCHK(clSetKernelArg(kernels[n], 8, sizeof(cl_uint), &pad1_r));
    
        //// End Task 2 //////////////////////////////////////////////////////////

