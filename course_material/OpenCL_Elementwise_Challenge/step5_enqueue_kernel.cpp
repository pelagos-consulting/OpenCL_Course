    // Event for the kernel
    cl_event kernel_event;
    
    // Now enqueue the kernel
    H_ERRCHK(
        clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            work_dim,
            NULL,
            global_size,
            local_size,
            0,
            NULL,
            &kernel_event
        ) 
    );

    // Wait on the kernel to finish
    H_ERRCHK(clWaitForEvents(1, &kernel_event));
