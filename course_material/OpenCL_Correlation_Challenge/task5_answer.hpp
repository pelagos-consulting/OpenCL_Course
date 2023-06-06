            //// Begin Task 5 - Code to enqueue the kernel ///////////////////////////
            
            // Enqueue the kernel kernels[tid] using command_queues[tid]
            // work_dim, local_size, and global_size
            H_ERRCHK(clEnqueueNDRangeKernel(
                    command_queues[tid],
                    kernels[tid],
                    work_dim,
                    NULL,
                    global_size,
                    local_size,
                    0, 
                    NULL,
                    NULL
                ) 
            );
            
            //// End Task 5 ///////////////////////////////////////////////////////////

