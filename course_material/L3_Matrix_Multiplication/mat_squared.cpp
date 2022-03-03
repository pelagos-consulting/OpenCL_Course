/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Define the size of the arrays to be computed
#define NROWS_C 520
#define NCOLS_C 1032

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

int main(int argc, char** argv) {
    
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
    
    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_FALSE;
    
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
    cl_uint dev_index = 0;
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // C and D are of size (N0_C, N1_C)
    cl_uint N0_C = NROWS_C, N1_C = NCOLS_C;
    size_t nbytes_C;

    // Read the input data into arrays and sanity check
    cl_float* array_C = (cl_float*)h_read_binary("array_C.dat", &nbytes_C);

    // Sanity check on incoming data
    assert(nbytes_C==N0_C*N1_C*sizeof(cl_float));   
    
    // Make an array to store the result in array_D
    // array_D is of the same size
    cl_float* array_D = (cl_float*)calloc(nbytes_C, 1);
    
    // Make Buffers on the compute device for matrices C and D
    cl_mem buffer_C = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_C, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_C");
    
    cl_mem buffer_D = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_C, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_D");

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_mat_squared.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_squared", &errcode);
    h_errchk(errcode, "Creating Kernel");
    
    // Set arguments to the kernel (not thread safe)
    h_errchk(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_C ),
        "setting kernel argument 0"
    );
    h_errchk(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_D ),
        "setting kernel argument 1"
    );
    h_errchk(
        clSetKernelArg(kernel, 2, sizeof(cl_uint), &N0_C ),
        "setting kernel argument 2"
    );
    h_errchk(
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &N1_C ),
        "setting kernel argument 3"
    );

    // Write memory from the host
    // to buffer_C on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    h_errchk(
        clEnqueueWriteBuffer(command_queue,
                            buffer_C,
                            blocking,
                            0,
                            nbytes_C,
                            array_C,
                            0,
                            NULL,
                            NULL), 
        "Writing to buffer_C from host"
    );
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 16, 1 };
    
    // Desired global_size
    const size_t global_size[]={ N0_C, N1_C };
    
    // Enlarge the global size so that 
    // an integer number of local sizes fits within it
    h_fit_global_size(global_size, 
                      local_size, 
                      work_dim
    );
    
    // Event for the kernel
    cl_event kernel_event;
    
    // Now enqueue the kernel
    h_errchk(
        clEnqueueNDRangeKernel(command_queue,
                                kernel,
                                work_dim,
                                NULL,
                                global_size,
                                local_size,
                                0,
                                NULL,
                                &kernel_event), 
        "Running the kernel"
    );

    // Wait on the kernel to finish
    h_errchk(
        clWaitForEvents(1, &kernel_event),
        "Waiting on the kernel"
    );
    
    // Read memory from the buffer to the host
    h_errchk(
        clEnqueueReadBuffer(command_queue,
                            buffer_D,
                            blocking,
                            0,
                            nbytes_C,
                            array_D,
                            1,
                            &kernel_event,
                            NULL), 
             "Copying matrix D from device to host"
    );
    
    // Write out the result to file
    h_write_binary(array_C, "array_D.dat", nbytes_C);

    // Free the OpenCL buffers
    h_errchk(
        clReleaseMemObject(buffer_C),
        "releasing buffer C"
    );
    h_errchk(
        clReleaseMemObject(buffer_D),
        "releasing buffer D"
    );
    
    // Clean up memory that was allocated on the read   
    free(array_C);
    free(array_D);
    
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

