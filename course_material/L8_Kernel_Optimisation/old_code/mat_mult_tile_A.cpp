/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

//// Step 1. Setup headers and parse command line arguments ////

#include <cassert>
#include <cmath>
#include <iostream>

// Bring in the size of the matrices
#include "mat_size.hpp"

// Bring in the library to work with matrices
#include "mat_helper.hpp"

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

typedef cl_float float_type;

void prep_mat_kernel(cl_kernel kernel, 
                 size_t* local_size,
                 size_t* global_size,
                 size_t ndim,
                 void* data) {
                 
    size_t* nbytes_line=(size_t*)data;

    // Local size for shared_AT is going to be (local_size[1], chunk_len)
    H_ERRCHK(clSetKernelArg(kernel, 3, local_size[1]*(*nbytes_line), NULL ));
    
    // Local size for shared_B is going to be (local_size[0], chunk_len)
    H_ERRCHK(clSetKernelArg(kernel, 4, local_size[0]*(*nbytes_line), NULL ));
}

int main(int argc, char** argv) {

    // Parse arguments and set the target device
    cl_device_type target_device;
    cl_uint dev_index = h_parse_args(argc, argv, &target_device);
    
    // Useful for checking OpenCL errors
    cl_int errcode;

    // Create handles to platforms, 
    // devices, and contexts

    // Number of platforms discovered
    cl_uint num_platforms;

    // Number of devices discovered
    cl_uint num_devices;

    // Pointer to an array of platforms
    cl_platform_id *platforms = NULL;

    // Pointer to an array of devices
    cl_device_id *devices = NULL;

    // Pointer to an array of contexts
    cl_context *contexts = NULL;
    
    //// Step 2. Discover resources ////
    
    // Helper function to acquire devices
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);
    
    // Number of command queues to generate
    cl_uint num_command_queues = num_devices;
    
    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_TRUE;

    // Do we enable blocking IO?
    cl_bool blocking = CL_TRUE;
    
    //// Step 3. Allocate command queues and choose a compute device ////
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        ordering,
        profiling
    );

    // Choose the first available context
    // and compute device to use
    // Also make sure command line arguments are sane
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // A is of size (N0_C, N1_A)
    // AT is of size (N1_A, N0_C)
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)
    
    //// Step 4. Prepare matrices A, B, and C on the Host ////
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;

    // Number of chunks
    cl_uint chunk_len = 32;
    cl_uint nchunks = N1_A/chunk_len;
    // Make sure chunk length is sane
    assert(N1_A%chunk_len==0);

    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float_type);
    size_t nbytes_B = N1_A*N1_C*sizeof(float_type);
    size_t nbytes_C = N0_C*N1_C*sizeof(float_type);

    // Allocate memory for matrices A, B, and C on the host
    float_type* A_h = (float_type*)h_alloc(nbytes_A);
    float_type* B_h = (float_type*)h_alloc(nbytes_B);
    float_type* C_h = (float_type*)h_alloc(nbytes_C);

    // Fill A_h and B_h with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
        
    //// Step 5. Allocate OpenCL Buffers for matrices A, B, and C ////
    
    // Make A_d by copying from A_h
    cl_mem A_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_A, 
        (void*)A_h, 
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Create buffer AT in the normal manner
    cl_mem AT_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_A, 
        NULL, 
        &errcode
    );
    H_ERRCHK(errcode);
   
    // Make B_d using B_h as a backing store
    cl_mem B_d = clCreateBuffer(
            context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            nbytes_B, 
            (void*)B_h, 
            &errcode
    );
    H_ERRCHK(errcode);
   
    // Allocate C on the device
    cl_mem C_d = clCreateBuffer(
            context, 
            CL_MEM_READ_WRITE, 
            nbytes_C, 
            NULL, 
            &errcode
    );
    H_ERRCHK(errcode);

    //// Step 6. Build the program from source for the chosen compute device ////
    
    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_mat_mult.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
    
    //// Step 7. Create a kernel from the compiled program and set arguments ////
    
    // Event for querying kernels
    cl_event kernel_event;

    // Run the transpose kernel first
    size_t work_dim = 2; 

    // Desired local size for all transactions
    size_t local_size[]={ 8, 8, 1 };
   
    // Create and run the transpose kernel
    cl_kernel kernel_transp=clCreateKernel(program, "transpose", &errcode);
    H_ERRCHK(errcode);

    // Desired global_size
    const size_t global_size_transp[]={ N1_A, N0_C };
    h_fit_global_size(global_size_transp, 
                      local_size, 
                      work_dim
    );    
    
    // Set kernel arguments for the transpose kernel
    H_ERRCHK(clSetKernelArg(kernel_transp, 0, sizeof(cl_mem), &A_d ));
    H_ERRCHK(clSetKernelArg(kernel_transp, 1, sizeof(cl_mem), &AT_d ));
    H_ERRCHK(clSetKernelArg(kernel_transp, 2, sizeof(cl_uint), &N0_C ));
    H_ERRCHK(clSetKernelArg(kernel_transp, 3, sizeof(cl_uint), &N1_A )); 

    // Now enqueue the transpose kernel and time it
    H_ERRCHK(
        clEnqueueNDRangeKernel(
            command_queue,
            kernel_transp,
            work_dim,
            NULL,
            global_size_transp,
            local_size,
            0,
            NULL,
            &kernel_event
        )
    );

    // Time the transpose kernel
    cl_double transpose_ms = h_get_event_time_ms(
        &kernel_event,
        "transpose",
        NULL
    );
    
    // Create and run the matrix multiplication kernel
    cl_kernel kernel_mat_mult=clCreateKernel(
        program, 
        "mat_mult_tile_A_stride_copy", 
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Set arguments to the kernel (not thread safe)
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 0, sizeof(cl_mem), &AT_d));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 1, sizeof(cl_mem), &B_d));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 2, sizeof(cl_mem), &C_d));
    // prep_mat_kernel will fill in the other arguments
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 5, sizeof(cl_uint), &N1_A));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 6, sizeof(cl_uint), &N0_C));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 7, sizeof(cl_uint), &N1_C));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 8, sizeof(cl_uint), &nchunks));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 9, sizeof(cl_uint), &chunk_len));
    
    // Number of statistical runs to do per experiment 
    size_t nstats = 3;
    
    // Desired local size
    local_size[0] = 16;
    local_size[1] = 16;
    
    // Desired global_size for 
    size_t global_size[]={ N1_C, N0_C };

    // data for local memory preparation kernel
    size_t prep_data=chunk_len*sizeof(float_type);
    
    // Prepare local memory arguments for execution
    prep_mat_kernel(
        kernel_mat_mult, 
        local_size,
        global_size,
        work_dim,
        &prep_data
    );
    
    // Run the optimisation program to time the tile kernel
    h_optimise_local(
        argc,
        argv,
        command_queue,
        kernel_mat_mult,
        device,
        // Desired global size of the problem
        global_size,
        // Desired local_size of the problem, use NULL for defaults
        local_size,
        // Number of dimensions in the kernel
        work_dim,
        // Number of times to run the kernel per experiment
        nstats,
        // Any pre-existing timing results
        transpose_ms,
        // Function for prepping the kernel prior to execution
        prep_mat_kernel,
        &prep_data
    );
    
    // Time the transpose kernel again
    H_ERRCHK(
        clEnqueueNDRangeKernel(
            command_queue,
            kernel_transp,
            work_dim,
            NULL,
            global_size_transp,
            local_size,
            0,
            NULL,
            &kernel_event
        )
    );
    transpose_ms = h_get_event_time_ms(
        &kernel_event,
        "transpose again",
        NULL
    );        

    //// Step 10. Copy the Buffer for matrix C back to the host ////

    // Read memory from the buffer to the host
    H_ERRCHK(
        clEnqueueReadBuffer(
            command_queue,
            C_d,
            blocking,
            0,
            nbytes_C,
            C_h,
            0,
            NULL,
            NULL
        )
    );

    //// Step 11. Test the answer against a known solution
    //// And write the contents of the matrices out to disk
   
    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);

    //// Step 12. Clean up arrays and release resources
    
    // Free the OpenCL buffers
    H_ERRCHK(clReleaseMemObject(A_d));
    H_ERRCHK(clReleaseMemObject(AT_d));
    H_ERRCHK(clReleaseMemObject(B_d));
    H_ERRCHK(clReleaseMemObject(C_d));
    
    // Clean up memory that was allocated on the read   
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_answer_h);
    
    // Clean up command queues
    h_release_command_queues(
        command_queues, 
        num_command_queues
    );
    
    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms
    );
}

