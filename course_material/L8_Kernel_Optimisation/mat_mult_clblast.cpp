/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>

// Include the size of arrays to be computed
#include "mat_size.hpp"

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

// Bring in the matrix helper library
#include "mat_helper.hpp"

// Include the CLBLAST library
#include "clblast_c.h"

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
    
    //// Prepare matrices A, B, and C on the Host ////
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

    // Make A_d by copying from A_h
    cl_mem A_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_A, 
        (void*)A_h, 
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
   
    cl_mem C_d = clCreateBuffer(
            context, 
            CL_MEM_READ_WRITE, 
            nbytes_C, 
            NULL, 
            &errcode
    );
    H_ERRCHK(errcode);

    // Constants for multiplication
    const float alpha=1.0;
    const float beta=0.0;
	
	cl_event kernel_event;
    
    // Set up a run for clblast
    cl_int nexperiments=1;
    cl_int npoints=2;
    size_t nbytes_output = nexperiments*npoints*sizeof(cl_double);
    cl_double* output_local = (cl_double*)malloc(nbytes_output);    
    
    // Run the experiment nstats times
    const size_t nstats=NSTATS;
    cl_double times_ms[nstats] = {0};
    cl_double time_ms=0.0;
    cl_double avg_time_ms=0.0;
    cl_double max_time_ms=0.0;
    cl_int max_time_n = 0;
    
    // Run the CLBlast kernel nstats times and collect times
    for (int n=0; n<nstats; n++) {
        
        // Start the clock
        auto t1 = std::chrono::high_resolution_clock::now();
       
        CLBlastStatusCode status = CLBlastSgemm(
            // Choose row-major ordering
            CLBlastLayoutRowMajor,
            // Do we transpose A?
            CLBlastTransposeNo,
            // Do we transpose B?
            CLBlastTransposeNo,
            // Number of rows in C (rows in A) to compute
            (const size_t)NROWS_C,
            // Number of columns in C (columns in B) to compute
            (const size_t)NCOLS_C,
            // Number of columns in A (rows in B) to compute
            (const size_t)NCOLS_A,
            alpha,
            // Buffer, starting offset in elements, length of contiguous dimension
            A_d, 0, (const size_t)NCOLS_A,
            B_d, 0, (const size_t)NCOLS_C,
            beta,
            C_d, 0, (const size_t)NCOLS_C,
            &command_queue,
            &kernel_event
        );
        
        // Make sure the matrix multiplication ran successfully
        assert(status==CLBlastSuccess);
        
        // Wait for events to finish
        H_ERRCHK(clWaitForEvents(1, &kernel_event));
        
        // Stop the clock
        auto t2 = std::chrono::high_resolution_clock::now();
        
        cl_double time_ms = (cl_double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;
        
        if (time_ms > max_time_ms) {
            max_time_ms = time_ms;
            max_time_n = n;
        }
        
        times_ms[n]=time_ms;
        
        avg_time_ms+=time_ms;
    }

    // Read memory from the buffer to the host
    H_ERRCHK(
        clEnqueueReadBuffer(command_queue,
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
    
    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);

    // Calculate the mean and average times
    // Leave the longest time out of the calculation
    avg_time_ms = avg_time_ms - max_time_ms;
    avg_time_ms/=(cl_double)(nstats-1);
    cl_double std_time_ms=0.0, scratch=0.0;
    
    for (int n=0; n<nstats; n++) {
        scratch=times_ms[n]-avg_time_ms;
        if (n!=max_time_n) {
            std_time_ms+=(scratch*scratch);
        }
    }
    std_time_ms=sqrt(std_time_ms)/(cl_double)(nstats-1);
    
    output_local[0]=avg_time_ms;
    output_local[1]=std_time_ms;
    
    h_write_binary(output_local, "output_local.dat", nbytes_output);
    free(output_local);

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

    return 0;
}

