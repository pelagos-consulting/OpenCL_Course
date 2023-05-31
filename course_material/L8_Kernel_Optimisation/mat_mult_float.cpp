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
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)
    
    //// Step 4. Prepare matrices A, B, and C on the Host ////
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;

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
    
    // Make B_d using a copy from B_h
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
    
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_mult_float", &errcode);
    H_ERRCHK(errcode);
    
    // Set arguments to the kernel (not thread safe)
    H_ERRCHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_d));
    H_ERRCHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_d));
    H_ERRCHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_d));
    H_ERRCHK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &N1_A));
    H_ERRCHK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &N0_C));
    H_ERRCHK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &N1_C));

    //// Step 8. Upload matrices ////
    //// We don't need to upload matrices ////
    //// because we copied at buffer creation ////

    
    //// Step 9. Run the kernel to compute C from A and B ////

    // Number of dimensions in the kernel
    size_t work_dim = 2;
    
    // Number of statistical runs to do per experiment 
    size_t nstats = NSTATS;
    
    // Desired local size
    size_t local_size[]={ 8, 8 };
    
    // Desired global_size
    size_t global_size[]={ N1_C, N0_C };
    
    // Run the optimisation program
    h_optimise_local(
        argc,
        argv,
        command_queue,
        kernel,
        device,
        global_size,
        local_size,
        work_dim,
        nstats,
        0,
        NULL,
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

