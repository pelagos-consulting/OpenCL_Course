/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <chrono>
#include <iostream>

// Define the size of the arrays to be computed
#define NCOLS_A 1024
#define NROWS_C 1024
#define NCOLS_C 1024

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

int main(int argc, char**argv) {
    // Start the clock
    auto time1 = std::chrono::high_resolution_clock::now();
    
    // Useful for checking OpenCL errors
    cl_int errcode;

    // Create handles to platforms, devices, and contexts
    cl_uint num_platforms;
    cl_uint num_devices;
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;
    cl_context *contexts = NULL;

    // Discover platforms and devices and create contexts
    cl_device_type target_device=CL_DEVICE_TYPE_ALL;
    
    // Helper function to acquire devices
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);
    
    // Number of command queues to generate
    cl_uint num_command_queues = num_devices;
    
    // Allocate command queues
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        CL_FALSE,
        CL_FALSE
    );

    // Choose the first available context and compute device to use
    cl_uint dev_index = 0;
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
    
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    size_t nbytes_A, nbytes_B, nbytes_C;

    // Read the input data into arrays and sanity check
    cl_float* array_A = (cl_float*)h_read_binary("array_A.dat", &nbytes_A);
    cl_float* array_B = (cl_float*)h_read_binary("array_B.dat", &nbytes_B);

    // Sanity check on incoming data
    assert(nbytes_A==N0_C*N1_A*sizeof(cl_float));   
    assert(nbytes_B==N1_A*N1_C*sizeof(cl_float));
    nbytes_C=N0_C*N1_C*sizeof(cl_float);
    
    // Make an array to store the result in array_C
    cl_float* array_C = (cl_float*)calloc(nbytes_C, 1);
    
    // Make buffers for bringing data in and out of the computation
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_A, NULL, &errcode);
    h_errchk(errcode, "Creating buffer_A");
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_B, NULL, &errcode);
    h_errchk(errcode, "Creating buffer_B");
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_C, NULL, &errcode);
    h_errchk(errcode, "Creating buffer_C");

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary("kernels_mat_mult.c", &nbytes_src);

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_mult", &errcode);
    h_errchk(errcode, "Creating Kernel");
    
    // Now run the kernel
    
    // Set arguments to the kernel
    h_errchk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A ),"setting kernel argument 0");
    h_errchk(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B ),"setting kernel argument 1");
    h_errchk(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C ),"setting kernel argument 2");
    h_errchk(clSetKernelArg(kernel, 3, sizeof(cl_uint), &N1_A ),"setting kernel argument 3");
    h_errchk(clSetKernelArg(kernel, 4, sizeof(cl_uint), &N0_C ),"setting kernel argument 4");
    h_errchk(clSetKernelArg(kernel, 5, sizeof(cl_uint), &N1_C ),"setting kernel argument 5");

    // Write memory to buffer_A and buffer_B from the host
    h_errchk(clEnqueueWriteBuffer(command_queue,
                            buffer_A,
                            CL_TRUE,
                            0,
                            nbytes_A,
                            array_A,
                            0,
                            NULL,
                            NULL), "Writing to buffer_A from host");

    h_errchk(clEnqueueWriteBuffer(command_queue,
                            buffer_B,
                            CL_TRUE,
                            0,
                            nbytes_B,
                            array_B,
                            0,
                            NULL,
                            NULL), "Writing to buffer_B from host");
    
    // Number of dimensions in the kernel
    cl_uint work_dim=2;
    const size_t global_work_size[]={ N0_C, N1_C };
    const size_t local_work_size[]={ 16, 1 };
    cl_event kernel_event;
    
    // Now enqueue the kernel
    h_errchk(clEnqueueNDRangeKernel(command_queue,
                                    kernel,
                                    work_dim,
                                    NULL,
                                    global_work_size,
                                    local_work_size,
                                    0,
                                    NULL,
                                    &kernel_event), "Running the kernel");

    // Read memory from the buffer to the host
    h_errchk(clEnqueueReadBuffer(command_queue,
                            buffer_C,
                            CL_TRUE,
                            0,
                            nbytes_C,
                            array_C,
                            1,
                            &kernel_event,
                            NULL), "Copying matrix C from device to host");

    // Write out the result to file
    h_write_binary(array_C, "array_C.dat", nbytes_C);
    
    // Read the answer from disk
    cl_float* array_C_answer = (cl_float*)h_read_binary("array_C_answer.dat", &nbytes_C);

    // Check the difference between the original and the computed matrix product
    // using the Root Mean Squared indicator
    cl_long nelements = (cl_long)N0_C*(cl_long)N1_C;
    cl_double rms=0.0;
    for (int i=0; i<nelements; i++ ) {
        rms+=(array_C[i]-array_C_answer[i])*(array_C[i]-array_C_answer[i]);
    }
    rms/=(cl_double)nelements;
    rms=sqrt(rms);
    printf("RMS difference is %g\n", rms);

    // Clean up memory that was allocated on the read   
    free(array_A);
    free(array_B);
    free(array_C);
    free(array_C_answer);
    
    // Clean up command queues
    h_release_command_queues(command_queues, num_command_queues);
    
    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms);

    // Stop the clock
    auto time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<cl_double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<cl_double>>(time2-time1);
    std::cout << "Elapsed time is " << elapsed_time.count() << "seconds" << std::endl;
}

