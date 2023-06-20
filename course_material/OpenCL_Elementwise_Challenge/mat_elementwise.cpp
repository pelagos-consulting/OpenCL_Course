/* Code to perform Hadamard (elementwise) multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Define the size of the arrays to be computed
#define NROWS_F 8
#define NCOLS_F 4

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

// Bring in helper header to work with matrices
#include "mat_helper.hpp"

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
    
    // D, E, F is of size (N0_F, N1_F)
    cl_uint N0_F = NROWS_F, N1_F = NCOLS_F;

    // Number of bytes in each matrix
    size_t nbytes_D=N0_F*N1_F*sizeof(float);   
    size_t nbytes_E=N0_F*N1_F*sizeof(float);
    size_t nbytes_F=N0_F*N1_F*sizeof(float);

    // Allocate memory for matrices A, B, and C on the host
    cl_float* D_h = (cl_float*)h_alloc(nbytes_D);
    cl_float* E_h = (cl_float*)h_alloc(nbytes_E);
    cl_float* F_h = (cl_float*)h_alloc(nbytes_F);

    // Fill host matrices with random numbers in the range 0, 1
    m_random(D_h, N0_F, N1_F);
    m_random(E_h, N0_F, N1_F);

    //// Step 2. Use clCreateBuffer to allocate OpenCL buffers
    //// for arrays D_d, E_d, and F_d. ////

    // Uncomment for the shortcut answer
    // #include "step2_create_buffers.cpp"

    
    
    
    
    //// End code: ////

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_elementwise.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);

    //// Step 3. Create the kernel and set kernel arguments ////
    //// Use clCreateKernel to create a kernel from program
    //// Use clSetKernelArg to set kernel arguments ////
    //// Select the kernel called mat_elementwise ////

    // Uncomment for the shortcut answer
    // #include "step3_create_kernel.cpp"

    
    
    
    
    //// End code: ////

    // Write memory from the host
    // to D_d and E_d on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    //// Step 4. Copy D_h and E_h to Buffers D_d and E_d
    //// Use clEnqueueWriteBuffer to copy memory ////
    //// from the host to the buffer ////
    
    // Uncomment for the shortcut answer
    // #include "step4_write_buffers.cpp"
    
    
    
    
    
    //// End code: ////
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 2, 2 };
    
    // Desired global_size
    const size_t global_size[]={ N1_F, N0_F };
    
    // Enlarge the global size so that 
    // an integer number of local sizes fits within it
    h_fit_global_size(global_size, 
                      local_size, 
                      work_dim
    );
    
    //// Step 5. Enqueue the kernel and wait for it to finish ////
    //// Create a cl_event called kernel_event ////
    //// Use clEnqueueNDRangeKernel to enqueue the kernel ////
    //// Use command_queue, kernel, work_dim, global_size, ////
    //// local_size, and kernel_event as parameters ////
    //// Use clWaitForEvents //// to wait on kernel_event

    // Uncomment for the shortcut answer
    // #include "step5_enqueue_kernel.cpp"
    
    
    
    
    
    //// End code: ////

    //// Step 6. Read memory from F_d back to F_h on the host ////
    //// Use clEnqueueReadBuffer to perform the copy

    // Uncomment for the shortcut answer
    // #include "step6_read_buffer.cpp"

    
    
    
    
    //// End code: ////

    // Check the answer against a known solution
    float* F_answer_h = (float*)calloc(nbytes_F, 1);
    float* F_residual_h = (float*)calloc(nbytes_F, 1);

    // Compute the known solution
    m_hadamard(D_h, E_h, F_answer_h, N0_F, N1_F);

    // Compute the residual between F_h and F_answer_h
    m_residual(F_answer_h, F_h, F_residual_h, N0_F, N1_F);

    // Pretty print the output matrices
    std::cout << "The output array F_h (as computed with OpenCL) is\n";
    m_show_matrix(F_h, N0_F, N1_F);

    std::cout << "The CPU solution (F_answer_h) is \n";
    m_show_matrix(F_answer_h, N0_F, N1_F);
    
    std::cout << "The residual (F_answer_h-F_h) is\n";
    m_show_matrix(F_residual_h, N0_F, N1_F);

    // Print the maximum error between matrices
    float max_err = m_max_error(F_h, F_answer_h, N0_F, N1_F);

    // Write out the result to file
    h_write_binary(D_h, "array_D.dat", nbytes_D);
    h_write_binary(E_h, "array_E.dat", nbytes_E);
    h_write_binary(F_h, "array_F.dat", nbytes_F);

    //// Step 7. Free the OpenCL buffers D_d, E_d, and F_d ////
    //// Use clReleaseMemObject to free the buffers ////

    // Uncomment for the shortcut answer
    // #include "step7_release_memobjects.cpp"
    
    
    
    
    
    //// End code: ////

    // Clean up memory that was allocated on the host 
    free(D_h);
    free(E_h);
    free(F_h);
    free(F_answer_h);
    free(F_residual_h);
    
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

