    H_ERRCHK(
        clEnqueueReadBuffer(
            command_queue,
            F_d,
            blocking,
            0,
            nbytes_F,
            F_h,
            1,
            &kernel_event,
            NULL
        ) 
    );