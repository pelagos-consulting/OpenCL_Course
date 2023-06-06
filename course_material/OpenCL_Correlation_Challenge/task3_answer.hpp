            //// Begin Task 3 - Code to upload memory to the compute device buffer ////
            
            // Upload memory from images_in at offset
            // To srcs_d[tid], using command_queues[tid]
            H_ERRCHK(clEnqueueWriteBuffer(
                    command_queues[tid],
                    srcs_d[tid],
                    blocking,
                    0,
                    nbytes_image,
                    &images_in[offset],
                    0,
                    NULL,
                    NULL
                ) 
            );

            //// End Task 3 ///////////////////////////////////////////////////////////

