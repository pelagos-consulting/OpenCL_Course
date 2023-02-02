    // Make Buffers on the compute device for matrices D, E, and F
    cl_mem D_d = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_D, 
                                     NULL, 
                                     &errcode);
    H_ERRCHK(errcode);
    
    cl_mem E_d = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_E, 
                                     NULL, 
                                     &errcode);
    H_ERRCHK(errcode);
    
    cl_mem F_d = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_F, 
                                     NULL, 
                                     &errcode);
    H_ERRCHK(errcode);

