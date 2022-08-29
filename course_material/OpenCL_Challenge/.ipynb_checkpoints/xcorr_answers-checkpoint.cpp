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

    // Create memory for images out
    size_t nbytes_output=NIMAGES*N0*N1*sizeof(float_type);
    float_type* images_out = (float_type*)h_alloc(nbytes_output);
    
    // Assume that images_in will have dimensions (NIMAGES, N0, N1) and will have row-major ordering

    // Read in images
    size_t nbytes;
    float_type* images_in = (float_type*)h_read_file("images_in.dat", "rb", &nbytes);
    assert(nbytes == NIMAGES*N0*N1*sizeof(float_type));

    // Read in image Kernel
    size_t nelements_image_kernel = (L0+R0+1)*(L1+R1+1);
    float_type* image_kernel = (float_type*)h_read_file("image_kernel.dat", "rb", &nbytes);
    assert(nbytes == nelements_image_kernel*sizeof(float_type));

    // Read kernel sources 
    const char* filename = "kernels.cl";
    char* source = (char*)h_read_file(filename, "r", &nbytes);

    // Create Programs and kernels using this source
    cl_program *programs = (cl_program*)calloc(num_devices, sizeof(cl_program));
    cl_kernel *kernels = (cl_kernel*)calloc(num_devices, sizeof(cl_kernel));
    
    for (cl_uint n=0; n<num_devices; n++) {
        // Make the program from source
        programs[n] = h_build_program(source, contexts[n], devices[n]);
        // And make the kernel
        kernels[n] = clCreateKernel(programs[n], "xcorr", &ret_code);
        h_errchk(ret_code, "Making a kernel");
    }

    // Create memory for images in and images out for each device
    cl_mem *buffer_srces = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *buffer_dests = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *buffer_kerns = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
   
    // Create buffers
    for (cl_uint n=0; n<num_devices; n++) {
        // Create buffers for sources
        buffer_srces[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                N0*N1*sizeof(float_type),
                NULL,
                &ret_code);
        h_errchk(ret_code, "Creating buffers for sources");

        // Create buffers for destination
        buffer_dests[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                N0*N1*sizeof(float_type),
                NULL,
                &ret_code);
        h_errchk(ret_code, "Creating buffers for destinations");

        // Copy host memory for the image kernel
        buffer_kerns[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_COPY_HOST_PTR,
                nelements_image_kernel*sizeof(float_type),
                (void*)image_kernel,
                &ret_code);
        h_errchk(ret_code, "Creating buffers for image kernel");

        // Just for kernel arguments
        cl_int len0_src = N0, len1_src = N1, pad0_l = L0, pad0_r = R0, pad1_l = L1, pad1_r = R1;

        // Set kernel arguments here for convenience
        h_errchk(clSetKernelArg(kernels[n], 0, sizeof(buffer_srces[n]), &buffer_srces[n]), "Set kernel argument 0");
        h_errchk(clSetKernelArg(kernels[n], 1, sizeof(buffer_dests[n]), &buffer_dests[n]), "Set kernel argument 1");
        h_errchk(clSetKernelArg(kernels[n], 2, sizeof(buffer_kerns[n]), &buffer_kerns[n]), "Set kernel argument 2");
        h_errchk(clSetKernelArg(kernels[n], 3, sizeof(cl_int), &len0_src),  "Set kernel argument 3");
        h_errchk(clSetKernelArg(kernels[n], 4, sizeof(cl_int), &len1_src),  "Set kernel argument 4");
        h_errchk(clSetKernelArg(kernels[n], 5, sizeof(cl_int), &pad0_l),    "Set kernel argument 5");
        h_errchk(clSetKernelArg(kernels[n], 6, sizeof(cl_int), &pad0_r),    "Set kernel argument 6");
        h_errchk(clSetKernelArg(kernels[n], 7, sizeof(cl_int), &pad1_l),    "Set kernel argument 7");
        h_errchk(clSetKernelArg(kernels[n], 8, sizeof(cl_int), &pad1_r),    "Set kernel argument 8");
    }

    // Use OpenMP to dynamically distribute threads across the available workflow of images
    //omp_set_dynamic(0);
    //omp_set_num_threads(num_devices);
    
    // This counter keeps track of images process by all iterations
    auto t1 = std::chrono::high_resolution_clock::now();
    
    cl_uint* it_count = (cl_uint*)calloc(num_devices, sizeof(cl_uint)); 
    
    // Enqueue the kernel
    cl_uint work_dim2 = 2;
            
    // Desired local size
    const size_t local_size[]={ 16, 16 };
    
    // Fit the desired global_size
    const size_t global_size[]={ N0, N1 };
    h_fit_global_size(global_size, local_size, work_dim);

    for (cl_uint i = 0; i<NITERS; i++) {
        printf("Processing iteration %d of %d\n", i+1, NITERS);
        
        #pragma omp parallel for default(none) schedule(dynamic, 1) num_threads(num_devices) \
            shared(images_in, buffer_dests, buffer_srces, \
                    images_out, image_kernel, nelements_image_kernel, \
                    command_queues, kernels, buffer_kerns, it_count)
        for (cl_uint n=0; n<NIMAGES; n++) {
            // Get the thread_id
            int tid = omp_get_thread_num();
            it_count[tid] += 1;
            
            // Load memory from images in using the offset
            size_t offset = n*N0*N1;

            //printf("Processing image %d of %d with device %d\n", n+1, NIMAGES, tid);
            
            // Write from main memory to the source buffer
            h_errchk(clEnqueueWriteBuffer(
                        command_queues[tid],
                        buffer_srces[tid],
                        blocking,
                        0,
                        N0*N1*sizeof(float_type),
                        images_in + offset,
                        0,
                        NULL,
                        NULL), "Writing to source buffer");
            
            // Upload the images kernel
            h_errchk(clEnqueueWriteBuffer(
                        command_queues[tid],
                        buffer_kerns[tid],
                        blocking,
                        0,
                        nelements_image_kernel*sizeof(float_type),
                        image_kernel,
                        0,
                        NULL,
                        NULL), "Writing to image kernel buffer");

            


            // Enqueue the kernel
            h_errchk(clEnqueueNDRangeKernel(
                        command_queues[tid],
                        kernels[tid],
                        work_dims,
                        NULL,
                        global_size,
                        local_size,
                        0, 
                        NULL,
                        NULL), "Running the xcorr kernel");

            // Read from the buffer to main memory and block
            h_errchk(clEnqueueReadBuffer(
                        command_queues[tid],
                        buffer_dests[tid],
                        blocking,
                        0,
                        N0*N1*sizeof(float_type),
                        images_out + offset,
                        0,
                        NULL,
                        NULL), "Writing to buffer");
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    double duration = time_span.count();
    
    cl_uint num_images = NITERS*NIMAGES;
    for (cl_uint i = 0; i< num_devices; i++) {
        //h_report_on_device(devices[i]);
        float_type pct = 100*(float_type)it_count[i]/(float_type)num_images;
        printf("Device %d processed %d of %d images (%0.2f\%)\n", i, it_count[i], num_images, pct);
    }
    printf("Overall processing rate %0.2f images/s\n", (double)num_images/duration);

    // Write output data to output file
    h_write_binary(images_out, "images_out.dat", )
    
    // Free allocated memory
    free(source);
    free(image_kernel);
    free(images_in);
    free(images_out);
    free(it_count);

    // Release command queues
    h_release_command_queues(command_queues, num_command_queues);

    // Release programs and kernels
    for (cl_uint n=0; n<num_devices; n++) {
        h_errchk(clReleaseKernel(kernels[n]), "Releasing kernel");
        h_errchk(clReleaseProgram(programs[n]), "Releasing program");
        h_errchk(clReleaseMemObject(buffer_srces[n]),"Releasing sources buffer");
        h_errchk(clReleaseMemObject(buffer_dests[n]),"Releasing dests buffer");
        h_errchk(clReleaseMemObject(buffer_kerns[n]),"Releasing image kernels buffer");
    }

    // Free memory
    free(buffer_srces);
    free(buffer_dests);
    free(buffer_kerns);
    free(programs);
    free(kernels);

    // Release devices and contexts
    h_release_devices(devices, num_devices, contexts, platforms);
}
