// Testbench for cross-correlation algorithm
#include <assert.h>
#include <cstdio>
#include <cstdint>
#include <omp.h>
#include <cmath>
#include <chrono>

// Include helper files
#include "cl_helper.hpp"
#include "mat_helper.hpp"

// Image sizes
#define N0 5
#define N1 4

// Amounts to pad
#define L0 0 
#define R0 2
#define L1 0
#define R1 2

typedef float float_type;

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
    
    // Do we enable blocking on copies
    cl_bool blocking = CL_TRUE;

    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_FALSE;

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
    
    // Size of the kernel
    const size_t K0=L0+R0+1;
    const size_t K1=L1+R1+1;

    // Make the image kernel
    float_type image_kern[K0*K1] = {-1,-1,-1,\
                                -1, 8,-1,\
                                -1,-1,-1};
    
    // Number of bytes for a single image
    size_t nbytes_image = N0*N1*sizeof(float_type);
    size_t nbytes_kern = K0*K1*sizeof(float_type);    
    
    // Allocate storage for the input and fill
    // with pseudorandom numbers
    float_type* image_in = (float_type*)h_alloc(nbytes_image); 
    m_random(image_in, N0, N1);
    
    // scale up random numbers for debugging purposes
    for (int i=0; i<N0*N1; i++) {
        image_in[i] = round(9.0*image_in[i]);
    }
    
    // Allocate storage for the output 
    float_type* image_out = (float_type*)h_alloc(nbytes_image);
 
    // Allocate storage for the test image
    float_type* image_test = (float_type*)h_alloc(nbytes_image);
    
    // Allocate memory for the source image and copy from host
    cl_mem src_d = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        nbytes_image,
        image_in,
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Allocate memory for the image kernel and copy from host
    cl_mem krn_d = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        nbytes_kern,
        image_kern,
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Allocate memory for the output
    cl_mem dst_d = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        nbytes_image,
        NULL,
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Use clEnqueueFillBuffer to fill dst_d with zeros
    float_type zero = 0.0f;
    H_ERRCHK(
        clEnqueueFillBuffer(
            command_queue,
            dst_d,
            &zero,
            sizeof(float_type),
            0,
            nbytes_image,
            0,
            NULL,
            NULL
        )
    );    
            
    // Make a kernel
    size_t nbytes=0;
    const char* filename = "kernels.c";
    char* kernel_source = (char*)h_read_binary(filename, &nbytes);
    const char* compiler_options = "";
    cl_program program = h_build_program(
        kernel_source, context, device, compiler_options
    );
    cl_kernel kernel = clCreateKernel(program, "xcorr", &errcode);
    H_ERRCHK(errcode);
    
    // Just for kernel arguments
    cl_uint len0_src = N0, len1_src = N1;
    cl_uint pad0_l = L0, pad0_r = R0, pad1_l = L1, pad1_r = R1;
    
    // Set kernel arguments here for convenience
    H_ERRCHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_d));
    H_ERRCHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_d));
    H_ERRCHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &krn_d));
    H_ERRCHK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &len0_src));
    H_ERRCHK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &len1_src));
    H_ERRCHK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &pad0_l));
    H_ERRCHK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &pad0_r));
    H_ERRCHK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &pad1_l));
    H_ERRCHK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &pad1_r));
     
    // Make up the local and global sizes to use
    cl_uint work_dim = 2;
    // Desired local size
    const size_t local_size[]={ 2, 2, 1 };
    // Fit the desired global_size
    const size_t global_size[]={ N1, N0 };
    h_fit_global_size(global_size, local_size, work_dim);
    
    // Run the kernel
    H_ERRCHK(clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            work_dim,
            NULL,
            global_size,
            local_size,
            0, 
            NULL,
            NULL
        ) 
    );
        
    // Read memory back out
    H_ERRCHK(clEnqueueReadBuffer(
            command_queue,
            dst_d,
            blocking,
            0,
            nbytes_image,
            (void*)image_out,
            0,
            NULL,
            NULL) 
    );

    // Perform the test correlation
    m_xcorr(
        image_test, 
        image_in, 
        image_kern,
        len0_src, len1_src, 
        pad0_l, pad0_r, 
        pad1_l, pad1_r
    );
    
    // Pretty print the matrices
    std::cout << "Image input" << "\n";
    m_show_matrix(image_in, len0_src, len1_src);    
    
    std::cout << "Image CPU" << "\n";
    m_show_matrix(image_test, len0_src, len1_src);
    
    std::cout << "Image OpenCL" << "\n";
    m_show_matrix(image_out, len0_src, len1_src);
    
    // Get the maximum error between the two
    m_max_error(image_test, image_out, len0_src, len1_src);

    std::cout << "Image kernel" << "\n";
    m_show_matrix(image_kern, K0, K1);  

    // Write output data to output file
    h_write_binary(image_out, "image_out.dat", nbytes_image);
    
    // Write kernel image to output file
    h_write_binary(image_kern, "image_kernel.dat", nbytes_kern);

    // Free memory
    free(image_in);
    free(image_out);
    free(image_test);
    free(kernel_source);
    
    // Release command queues
    h_release_command_queues(command_queues, num_command_queues);
    
    // Release programs, kernels and buffers
    H_ERRCHK(clReleaseKernel(kernel));
    H_ERRCHK(clReleaseProgram(program));
    H_ERRCHK(clReleaseMemObject(src_d));
    H_ERRCHK(clReleaseMemObject(dst_d));
    H_ERRCHK(clReleaseMemObject(krn_d));    
    
    // Release devices and contexts
    h_release_devices(devices, num_devices, contexts, platforms);
}
