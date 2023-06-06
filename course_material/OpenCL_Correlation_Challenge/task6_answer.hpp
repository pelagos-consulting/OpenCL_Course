            //// Begin Task 6 - Code to download memory from the compute device buffer
            
            //// Download memory buffers_dests[tid] to hosts allocation
            //// images_out at offset
            H_ERRCHK(clEnqueueReadBuffer(
                    command_queues[tid],
                    dsts_d[tid],
                    blocking,
                    0,
                    nbytes_image,
                    &images_out[offset],
                    0,
                    NULL,
                    NULL
                ) 
            );
            
            //// End Task 6 ///////////////////////////////////////////////////////////

