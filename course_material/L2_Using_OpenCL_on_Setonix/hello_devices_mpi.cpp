///
/// @file  hello_devices_mpi.cpp
/// 
/// @brief OpenCL function for demonstrating the use of MPI with OpenCL.
///
/// Written by Dr. Toby Potter 
/// for the Commonwealth Scientific and Industrial Research Organisation of Australia (CSIRO).
///

#include "cl_helper.hpp"
#include "mpi.h"

// Length of the vector
#define N0_A 512

// The kernel source. We use C++ raw literals to contain the kernel.
const char* kernel_source = R"(
__kernel void fill (__global float* A,  
                    float fill_value, 
                    unsigned int N) { 
            
    // A is of size (N,)
    size_t i0 = get_global_id(0);
    
    if (i0<N) {
        A[i0]=fill_value;
    }
}       
)";

// Simple kernel to fill a vector 

// Main program
int main(int argc, char** argv) {
    
    // Initialise MPI
    int ierr = MPI_Init (&argc, &argv);
    assert(ierr==0);
    
    // Get the number of ranks
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    // Get the MPI rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Print which rank we are using
    std::cout << "MPI rank " << rank << " of " << nranks << "\n";
 
    // Create an errorcode to store result from application runs
    cl_int errcode=CL_SUCCESS;
    
    // Set the target device
    cl_device_type target_device=CL_DEVICE_TYPE_GPU;
    
    //// Step 2. Discover resources ////
    cl_uint num_platforms=0, num_devices=0;
    cl_platform_id *platforms=NULL;
    cl_device_id *devices=NULL;
    cl_context *contexts=NULL;
    
    // Helper function to acquire devices
    // Create one context for every device
    // return flat arrays of all platforms, all devices,
    // and all contexts (contexts are same length as devices)
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);

    // Do we enable out-of-order execution in the command queues
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling in the command queues
    cl_bool profiling = CL_TRUE;
    
    // Create the command queues, one command queue
    // for every compute device
    cl_uint num_command_queues = num_devices;
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        ordering,
        profiling
    );
    
    // Choose the device index to use
    // based on rank and number of compute devices
    cl_uint dev_index = rank%num_devices;
    
    // Choose the context, device and command queue
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on available devices found
    for (cl_uint n=0; n<num_devices; n++) {
        std::cout << "device " << n << " of " << num_devices << std::endl;
        h_report_on_device(devices[n]);
    }
    
    // Allocate memory on the compute device for vector A
    size_t nbytes_A = N0_A*sizeof(float);
    cl_mem A_d = clCreateBuffer(
            context, 
            CL_MEM_READ_WRITE, 
            nbytes_A, 
            NULL, 
            &errcode
    );
    H_ERRCHK(errcode);
    
    // Allocate memory on the host for vector A
    float* A_h = (float*)calloc(nbytes_A, 1);
    
    // Compile the kernel source into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
    
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "fill", &errcode);
    H_ERRCHK(errcode);
    
    // Set kernel arguments
    cl_uint N=N0_A;
    cl_float fill_value=1.0;
    
    H_ERRCHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_d));
    H_ERRCHK(clSetKernelArg(kernel, 1, sizeof(float), &fill_value));
    H_ERRCHK(clSetKernelArg(kernel, 2, sizeof(cl_uint), &N));
    
    // Prepare grid sizes for the kernel
    size_t work_dim=1;
    const size_t local_size[] = { 64, 1, 1 };
    const size_t global_size[] = {N, 1, 1};
     
    // Enlarge the global size so that 
    // an integer number of local sizes fits within it
    h_fit_global_size(global_size, 
                      local_size, 
                      work_dim
    );
    
    // Event for the kernel
    cl_event kernel_event;
    
    // Now enqueue the kernel to the command queue
    H_ERRCHK(
        clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            work_dim,
            NULL,
            global_size,
            local_size,
            0,
            NULL,
            &kernel_event
        )
    );
    
    // Read memory from the buffer to the host
    
    // Is our command blocking?
    cl_bool blocking=CL_TRUE;
    
    H_ERRCHK(
        clEnqueueReadBuffer(
            command_queue,
            A_d,
            blocking,
            0,
            nbytes_A,
            A_h,
            1,
            &kernel_event,
            NULL
        )
    );
    
    // Free the OpenCL buffers
    H_ERRCHK(clReleaseMemObject(A_d));
    
    // Check the memory allocation to see if it was filled correctly
    for (cl_uint i0=0; i0<N0_A; i0++) {
        assert(A_h[i0]==fill_value);
    }
    
    // Free host memory
    free(A_h);
    
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
     
    // End the MPI application
    MPI_Finalize();
}
