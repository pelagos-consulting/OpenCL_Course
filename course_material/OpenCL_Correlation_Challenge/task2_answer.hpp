        //// Begin Task 1 - Code to create the OpenCL buffers for each thread ////
        
        // Fill srcs_d[n], dsts_d[n], kerns_d[n] 
        // with buffers created by clCreateBuffer 
        // Use the H_ERRCHK routine to check output
        
        // srcs_d[n] is of size nbytes_image
        // dsts_d[n] is of size nbytes_image
        // kerns_d[n] is of size nbytes_image_kernel
        
        // the array image_kernel contains the host-allocated 
        // memory for the image kernel
        
        // Create buffers for sources
        srcs_d[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                nbytes_image,
                NULL,
                &errcode);
        H_ERRCHK(errcode);

        // Create buffers for destination
        dsts_d[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                nbytes_image,
                NULL,
                &errcode);
        H_ERRCHK(errcode);
        
        // Zero out the contents of dsts_d[n]
        float_type zero=0.0;
        H_ERRCHK(clEnqueueFillBuffer(
                command_queues[n],
                dsts_d[n],
                &zero,
                sizeof(float_type),
                0,
                nbytes_image,
                0,
                NULL,
                NULL
            )
        );

        // Create buffer for the image kernel, copy from host memory image_kernel to fill this
        kerns_d[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_COPY_HOST_PTR,
                nbytes_image_kernel,
                (void*)image_kernel,
                &errcode);
        H_ERRCHK(errcode);

        //// End Task 1 //////////////////////////////////////////////////////////

