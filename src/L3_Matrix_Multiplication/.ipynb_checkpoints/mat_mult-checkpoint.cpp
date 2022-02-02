/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <chrono>
#include <iostream>

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

int main(int argc, char**argv) {
    // Use the Chrono namespace?
    using namespace std::chrono;

    // Start the clock
    high_resolution_clock::time_point time1 = high_resolution_clock::now();
    
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
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    size_t nrows=1024;
    size_t ncols=1024;
    size_t element_size=sizeof(float);
    size_t nelements=nrows*ncols;
    size_t nbytes;

    // Read the input data into arrays
    float* array_A = (float*)h_read_binary("array_A.dat", &nbytes);
    float* array_B = (float*)h_read_binary("array_B.dat", &nbytes);
    float* array_C_answer = (float*)h_read_binary("array_C_answer.dat", &nbytes);
    
    // Sanity check on incoming data
    assert(nbytes==nelements*sizeof(float));
    
    // Make an array to store the result in array_C
    float* array_C = (float*)calloc(nbytes, 1);
    
    // Make buffers for bringing data in and out of the computation
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &errcode);
    errchk(errcode, "Creating buffer_A");
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &errcode);
    errchk(errcode, "Creating buffer_B");
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &errcode);
    errchk(errcode, "Creating buffer_C");

    // Now specify the kernel source.
    const char* kernel_source="\n\
        // standard matrix multiply kernel \n\
        __kernel void mat_mult (    __global float* A, \n\
                                    __global float* B, \n\
                                    __global float* C, \n\
                                    int nrows_A, \n\
                                    int nrows_B) { \n\
            \n\
            // i0 and i1 represent the coordinates in C \n\
            // We assume Fortran ordering for the matrices \n\
            size_t i0=get_global_id(0); \n\
            size_t i1=get_global_id(1); \n\
            float temp=0.0; \n\
            // Loop over columns of A and rows of B \n\
            for (int n=0; n<nrows_B; n++) { \
                // C has the same number of rows as A, \n\
                // and the same number of columns as B \n\
                // i0 is the row index of A \n\
                // i1 is the column index of B \n\
                temp+=A[n*nrows_A+i0]*B[i1*nrows_B+n]; \n\
            } \n\
            // Number of rows in C is same as number of rows in A \n\
            C[i1*nrows_A+i0]=temp; \n\
        } \n\
    ";

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_mult", &errcode);
    errchk(errcode, "Creating Kernel");
    
    // Now run the kernel
    
    // Set arguments to the kernel
    errchk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A ),"setting kernel argument 0");
    errchk(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B ),"setting kernel argument 1");
    errchk(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C ),"setting kernel argument 2");
    errchk(clSetKernelArg(kernel, 3, sizeof(int), &nrows ),"setting kernel argument 3");
    errchk(clSetKernelArg(kernel, 4, sizeof(int), &nrows ),"setting kernel argument 4");

    // Write memory to buffer_A and buffer_B from the host
    errchk(clEnqueueWriteBuffer(command_queue,
                            buffer_A,
                            CL_TRUE,
                            0,
                            nbytes,
                            array_A,
                            0,
                            NULL,
                            NULL), "Writing to buffer_A from host");

    errchk(clEnqueueWriteBuffer(command_queue,
                            buffer_B,
                            CL_TRUE,
                            0,
                            nbytes,
                            array_B,
                            0,
                            NULL,
                            NULL), "Writing to buffer_B from host");
    
    // Number of dimensions in the kernel
    cl_uint work_dim=2;
    const size_t global_work_size[]={ nrows, ncols };
    const size_t local_work_size[]={ nrows, ncols };
    cl_event kernel_event;
    
    // Now enqueue the kernel
    errchk(clEnqueueNDRangeKernel(command_queue,
                                    kernel,
                                    work_dim,
                                    0,
                                    global_work_size,
                                    local_work_size,
                                    0,
                                    NULL,
                                    &kernel_event), "Running the kernel");

    // Read memory from the buffer to the host
    errchk(clEnqueueReadBuffer(command_queue,
                            buffer_C,
                            CL_TRUE,
                            0,
                            nbytes,
                            array_C,
                            1,
                            &kernel_event,
                            NULL), "Copying matrix C from device to host");

    // Write out the answer to file
    h_write_binary(array_C, "array_C.dat", nbytes);

    // Check the difference between the original and the computed matrix product
    // using the Root Mean Squared indicator
    double rms=0.0;
    for (int i=0; i<nelements; i++ ) {
        rms+=(array_C[i]-array_C_answer[i])*(array_C[i]-array_C_answer[i]);
    }
    rms/=nelements;
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
    high_resolution_clock::time_point time2 = high_resolution_clock::now();
    duration<double> elapsed_time = duration_cast<duration<double>>(time2-time1);
    std::cout << "Elapsed time is " << elapsed_time.count() << "seconds" << std::endl;
}

