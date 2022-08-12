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
    

    
    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_TRUE;

    // Do we enable blocking IO?
    cl_bool blocking = CL_TRUE;
    
    // Number of scratch buffers, must be at least 3
    const int nscratch=4;
    
    // Number of command queues to generate
    cl_uint num_command_queues = nscratch;
    
    // Choose the first available context
    // and compute device to use
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        &device,
        &context,
        (cl_uint)1,
        (cl_uint)num_command_queues,
        ordering,
        profiling
    );
    
    // Command queue to do temporary things
    cl_command_queue command_queue = command_queues[0];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    size_t nbytes_U;
    
    // Read in the velocity from disk and find the maximum
    float_type* array_V = (float_type*)h_read_binary("array_V.dat", &nbytes_U);
    assert(nbytes_U==N0*N1*sizeof(float_type));
    float_type Vmax = 0.0;
    for (size_t i=0; i<N0*N1; i++) {
        Vmax = (array_V[i]>Vmax) ? array_V[i] : Vmax;
    }

    // Make up the timestep using maximum velocity
    float_type dt = CFL*std::min(D0, D1)/Vmax;
    
    printf("dt=%f, Vmax=%f\n", dt, Vmax);
    
    // Use a grid crossing time at maximum velocity to get the number of timesteps
    int NT = (int)std::max(D0*N0, D1*N1)/(dt*Vmax);
    
    // Make up the output array
    size_t nbytes_out = NT*N0*N1*sizeof(cl_float);
    cl_float* array_out = (cl_float*)h_alloc(nbytes_out);
    
    // Make Buffers on the compute device for matrices U0, U1, U2, V
    
    // Read-only buffer for V
    cl_mem buffer_V = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        nbytes_U, 
        (void*)array_V, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_V");
    
    // Create scratch buffers for the computation
    cl_mem buffers_U[nscratch];
    for (int n=0; n<nscratch; n++) {
        buffers_U[n] = clCreateBuffer(
            context, 
            CL_MEM_ALLOC_HOST_PTR, 
            nbytes_U, 
            NULL, 
            &errcode
        );
        h_errchk(errcode, "Creating scratch buffer.");
        
        // Zero out buffers
        cl_float zero=0.0f;
        h_errchk(
            clEnqueueFillBuffer(
                command_queue,
                buffers_U[n],
                &zero,
                sizeof(cl_float),
                0,
                nbytes_U,
                0,
                NULL,
                NULL
            ),
            "Filling buffer with zeroes."
        );
    }
    
    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "wave2d_4o", &errcode);
    h_errchk(errcode, "Creating wave2d_4o Kernel");

    // Arguments for the kernel
    cl_uint N0_k=N0, N1_k=N1;
    cl_float dt2=dt*dt, inv_dx02=1.0/(D0*D0), inv_dx12=1.0/(D1*D1);
    
    // Set arguments to the kernel (not thread safe)
    h_errchk(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_V ),
        "setting kernel argument 3"
    );
    h_errchk(
        clSetKernelArg(kernel, 4, sizeof(cl_uint), &N0_k ),
        "setting kernel argument 4"
    );
    h_errchk(
        clSetKernelArg(kernel, 5, sizeof(cl_uint), &N1_k ),
        "setting kernel argument 5"
    );
    h_errchk(
        clSetKernelArg(kernel, 6, sizeof(cl_float), &dt2 ),
        "setting kernel argument 6"
    );
    h_errchk(
        clSetKernelArg(kernel, 7, sizeof(cl_float), &inv_dx02 ),
        "setting kernel argument 7"
    );
    h_errchk(
        clSetKernelArg(kernel, 8, sizeof(cl_float), &inv_dx12 ),
        "setting kernel argument 8"
    );
    
    // Number of dimensions in the kernel
    size_t work_dim = 2;
    
    // Desired local size
    const size_t local_size[]={ 64, 4 };
    
    // Desired global_size
    const size_t global_size[]={ N1, N0 };
    h_fit_global_size(global_size, local_size, work_dim);
    
    // Main loop
    cl_mem U0, U1, U2;
    
    for (int n=0; n<NT; n++) {
        // Get the wavefields
        U0 = buffers_U[n%nscratch];
        U1 = buffers_U[(n+1)%nscratch];
        U2 = buffers_U[(n+2)%nscratch];
        
        // Set kernel arguments
        h_errchk(
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &U0 ),
            "setting kernel argument 0"
        );
        h_errchk(
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &U1 ),
            "setting kernel argument 1"
        );
        h_errchk(
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &U2 ),
            "setting kernel argument 2"
        );
        
        // Fetch the command queue
        command_queue = command_queues[n%nscratch];
        
        // Wait for all previous commands to finish
        h_errchk(
            clFinish(command_queue),
            "Waiting for all previous things to finish"
        );
        
        // Enqueue the wave solver    
        h_errchk(
            clEnqueueNDRangeKernel(
                command_queue,
                kernel,
                work_dim,
                NULL,
                global_size,
                local_size,
                0,
                NULL,
                NULL), 
            "Running the kernel"
        );
          
        // Read memory from the buffer to the host in an asynchronous manner
        h_errchk(
            clEnqueueReadBuffer(
                command_queue,
                U2,
                CL_FALSE,
                0,
                nbytes_U,
                &array_out[n*N0*N1],
                0,
                NULL,
                NULL), 
            "Asynchronous copy from U2 on device to host"
        );
    }

    // Make sure all work is done
    for (int i=0; i<nscratch; i++) {
        clFinish(command_queues[i]);
    }
    
    // Write out the result to file
    h_write_binary(array_out, "array_out.dat", nbytes_out);

    // Free the OpenCL buffers
    h_errchk(
        clReleaseMemObject(buffer_V),
        "releasing buffer V"
    );
    for (int n=0; n<nscratch; n++) {
        h_errchk(
            clReleaseMemObject(buffers_U[n]),
            "Releasing scratch buffer"
        );
    }
    
    // Clean up memory that was allocated on the read   
    free(array_V);
    free(array_out);
    
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

