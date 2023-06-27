// Main

#include <assert.h>
#include "cl_helper.hpp"
#include <cstdio>
#include <omp.h>
#include <chrono>

#include "mat_size.hpp"

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
    
    // Make a command queue and report on devices
    for (cl_uint n=0; n<num_devices; n++) {
        h_report_on_device(devices[n]);
    }
   
    // Create command queues, one for each device 
    cl_uint num_command_queues = num_devices;
    cl_command_queue* command_queues = h_create_command_queues(
            devices,
            contexts,
            num_devices,
            num_command_queues,
            ordering,
            profiling);

    // Number of Bytes for a single image
    size_t nbytes_image = N0*N1*sizeof(float_type);

    // Number of Bytes for the stack of images
    size_t nbytes_input=NIMAGES*nbytes_image;
    // Output stack is the same size as the input
    size_t nbytes_output=nbytes_input;
    
    // Allocate storage for the output 
    float_type* images_out = (float_type*)h_alloc(nbytes_output);
    
    // Assume that images_in will have dimensions (NIMAGES, N0, N1) and will have row-major ordering
    size_t nbytes;
    
    // Read in the images
    float_type* images_in = (float_type*)h_read_binary("images_in.dat", &nbytes);
    assert(nbytes == nbytes_input);

    // Make up the image kernel
    const size_t K0=L0+R0+1;
    const size_t K1=L1+R1+1;
    size_t nbytes_image_kernel = K0*K1*sizeof(float_type);

    // Make the image kernel
    float_type image_kernel[K0*K1] = {-1,-1,-1,\
                                -1, 8,-1,\
                                -1,-1,-1};

    // Read kernel sources 
    const char* filename = "kernels_answers.c";
    char* kernel_source = (char*)h_read_binary(filename, &nbytes);

    // Create Programs and kernels for all devices 
    cl_program *programs = (cl_program*)calloc(num_devices, sizeof(cl_program));
    cl_kernel *kernels = (cl_kernel*)calloc(num_devices, sizeof(cl_kernel));
    
    const char* compiler_options = "";
    for (cl_uint n=0; n<num_devices; n++) {
        // Make the program from source
        programs[n] = h_build_program(kernel_source, contexts[n], devices[n], compiler_options);
        // And make the kernel
        kernels[n] = clCreateKernel(programs[n], "xcorr", &errcode);
        H_ERRCHK(errcode);
    }

    // Create OpenCL buffer for source, destination, and image kernel
    cl_mem *srcs_d = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *dsts_d = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *kerns_d = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
   
    // Create input buffers for every device
    for (cl_uint n=0; n<num_devices; n++) {
        
        //// Begin Task 2 - Code to create the OpenCL buffers for each thread ////
        
        // Fill srcs_d[n], dsts_d[n], kerns_d[n] 
        // with buffers created by clCreateBuffer 
        // Use the H_ERRCHK routine to check output
        
        // srcs_d[n] is of size nbytes_image
        // dsts_d[n] is of size nbytes_image
        // kerns_d[n] is of size nbytes_image_kernel
        
        // the array image_kernel contains the host-allocated 
        // memory for the image kernel
        
        // Uncomment the line below for the shortcut solution
        // #include task2_answer.hpp
        
        // Create buffers for sources
        srcs_d[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                nbytes_image,
                NULL,
                &errcode);
        H_ERRCHK(errcode);

        // Create buffers for destination
        dsts_d[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                nbytes_image,
                NULL,
                &errcode);
        H_ERRCHK(errcode);
        
        // Zero out the contents of dsts_d[n]
        float_type zero=0.0;
        H_ERRCHK(clEnqueueFillBuffer(
                command_queues[n],
                dsts_d[n],
                &zero,
                sizeof(float_type),
                0,
                nbytes_image,
                0,
                NULL,
                NULL
            )
        );

        // Create buffer for the image kernel, copy from host memory image_kernel to fill this
        kerns_d[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_COPY_HOST_PTR,
                nbytes_image_kernel,
                (void*)image_kernel,
                &errcode);
        H_ERRCHK(errcode);

        //// End Task 2 //////////////////////////////////////////////////////////
        
        // Just for kernel arguments
        cl_uint len0_src = N0, len1_src = N1, pad0_l = L0, pad0_r = R0, pad1_l = L1, pad1_r = R1;

        //// Begin Task 3 - Code to set kernel arguments for each thread /////////
        
        // Uncomment the line below for the shortcut solution
        // #include task3_answer.hpp
        
        // Set kernel arguments for kernels[n]
        H_ERRCHK(clSetKernelArg(kernels[n], 0, sizeof(cl_mem), &srcs_d[n]));
        H_ERRCHK(clSetKernelArg(kernels[n], 1, sizeof(cl_mem), &dsts_d[n]));
        H_ERRCHK(clSetKernelArg(kernels[n], 2, sizeof(cl_mem), &kerns_d[n]));
        H_ERRCHK(clSetKernelArg(kernels[n], 3, sizeof(cl_uint), &len0_src));
        H_ERRCHK(clSetKernelArg(kernels[n], 4, sizeof(cl_uint), &len1_src));
        H_ERRCHK(clSetKernelArg(kernels[n], 5, sizeof(cl_uint), &pad0_l));
        H_ERRCHK(clSetKernelArg(kernels[n], 6, sizeof(cl_uint), &pad0_r));
        H_ERRCHK(clSetKernelArg(kernels[n], 7, sizeof(cl_uint), &pad1_l));
        H_ERRCHK(clSetKernelArg(kernels[n], 8, sizeof(cl_uint), &pad1_r));
    
        //// End Task 3 //////////////////////////////////////////////////////////
    }
    
    // Start the timer
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Keep track of how many images each device processed
    cl_uint* it_count = (cl_uint*)calloc(num_devices, sizeof(cl_uint)); 
    
    // Make up the local and global sizes to use
    cl_uint work_dim = 2;
    // Desired local size
    const size_t local_size[]={ 16, 16 };
    // Fit the desired global_size
    const size_t global_size[]={ N1, N0 };
    h_fit_global_size(global_size, local_size, work_dim);

    for (cl_uint i = 0; i<NITERS; i++) {
        printf("Processing iteration %d of %d\n", i+1, NITERS);
        
        #pragma omp parallel for default(none) schedule(dynamic, 1) num_threads(num_devices) \
            shared(local_size, global_size, work_dim, images_in, images_out, \
                    dsts_d, srcs_d, nbytes_image, \
                    blocking, command_queues, kernels, it_count)
        for (cl_uint n=0; n<NIMAGES; n++) {
            // Get the thread_id
            int tid = omp_get_thread_num();
            
            // Increment image counter for this device
            it_count[tid] += 1;
            
            // Load memory from images in using the offset
            size_t offset = n*N0*N1;
            
            //// Begin Task 4 - Code to upload memory to the compute device buffer ////
            
            // Uncomment the line below for the shortcut solution
            // #include task4_answer.hpp

            // Upload memory from images_in at offset
            // To srcs_d[tid], using command_queues[tid]
            H_ERRCHK(clEnqueueWriteBuffer(
                    command_queues[tid],
                    srcs_d[tid],
                    blocking,
                    0,
                    nbytes_image,
                    &images_in[offset],
                    0,
                    NULL,
                    NULL
                ) 
            );

            //// End Task 4 ///////////////////////////////////////////////////////////
    
            
            //// Begin Task 5 - Code to enqueue the kernel ///////////////////////////
            
            // Uncomment the line below for the shortcut solution
            // #include task5_answer.hpp
            
            // Enqueue the kernel kernels[tid] using command_queues[tid]
            // work_dim, local_size, and global_size
            H_ERRCHK(clEnqueueNDRangeKernel(
                    command_queues[tid],
                    kernels[tid],
                    work_dim,
                    NULL,
                    global_size,
                    local_size,
                    0, 
                    NULL,
                    NULL
                ) 
            );
            
            //// End Task 5 ///////////////////////////////////////////////////////////
            
            //// Begin Task 6 - Code to download memory from the compute device buffer
            
            // Uncomment the line below for the shortcut solution
            // #include task6_answer.hpp

            //// Download memory buffers_dests[tid] to hosts allocation
            //// images_out at offset
            H_ERRCHK(clEnqueueReadBuffer(
                    command_queues[tid],
                    dsts_d[tid],
                    blocking,
                    0,
                    nbytes_image,
                    &images_out[offset],
                    0,
                    NULL,
                    NULL
                ) 
            );
            
            //// End Task 6 ///////////////////////////////////////////////////////////
        }
    }

    // Stop the timer
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    double duration = time_span.count();
    
    // Get some statistics on how 
    cl_uint num_images = NITERS*NIMAGES;
    for (cl_uint i = 0; i< num_devices; i++) {
        //h_report_on_device(devices[i]);
        float_type pct = 100*(float_type)it_count[i]/(float_type)num_images;
        printf("Device %d processed %d of %d images (%0.2f%%)\n", i, it_count[i], num_images, pct);
    }
    printf("Overall processing rate %0.2f images/s\n", (double)num_images/duration);

    // Write output data to output file
    h_write_binary(images_out, "images_out.dat", nbytes_output);
    
    // Free allocated memory
    free(kernel_source);
    free(images_in);
    free(images_out);
    free(it_count);

    // Release command queues
    h_release_command_queues(command_queues, num_command_queues);

    // Release programs and kernels
    for (cl_uint n=0; n<num_devices; n++) {
        H_ERRCHK(clReleaseKernel(kernels[n]));
        H_ERRCHK(clReleaseProgram(programs[n]));
        H_ERRCHK(clReleaseMemObject(srcs_d[n]));
        H_ERRCHK(clReleaseMemObject(dsts_d[n]));
        H_ERRCHK(clReleaseMemObject(kerns_d[n]));
    }

    // Free memory
    free(srcs_d);
    free(dsts_d);
    free(kerns_d);
    free(programs);
    free(kernels);

    // Release devices and contexts
    h_release_devices(devices, num_devices, contexts, platforms);
}
