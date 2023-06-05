/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>
#include "omp.h"

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

    // Allocate memory for buffers
    cl_mem* As_d = (cl_mem*)malloc(num_devices*sizeof(cl_mem));
    cl_mem* Bs_d = (cl_mem*)malloc(num_devices*sizeof(cl_mem));    
    cl_mem* Cs_d = (cl_mem*)malloc(num_devices*sizeof(cl_mem));      
    
    for (cl_uint n=0; n<num_devices; n++) {
        
        // Report on the device
        h_report_on_device(devices[n]);
        
        // Create buffers for A
        As_d[n] = clCreateBuffer(
            contexts[n], 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            nbytes_A, 
            (void*)A_h, 
            &errcode
        );
        H_ERRCHK(errcode);

        // Create buffers for B        
        Bs_d[n] = clCreateBuffer(
            contexts[n], 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            nbytes_B, 
            (void*)B_h, 
            &errcode
        );
        H_ERRCHK(errcode);

        // Create buffers for C
        Cs_d[n] = clCreateBuffer(
            contexts[n], 
            CL_MEM_READ_WRITE, 
            nbytes_C, 
            NULL, 
            &errcode
        );
        H_ERRCHK(errcode);        
    }

    // Constants for multiplication
    const float alpha=1.0;
    const float beta=0.0;
    
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
    
    // Number of domains along each dimension
    cl_uint D0=4;
    cl_uint D1=4;
    
    // Make maximum local domain sizes
    size_t L0=(size_t)ceil((double)N0_C/(double)D0);
    size_t L1=(size_t)ceil((double)N1_C/(double)D1);    
    
    // Set the number of OpenMP threads
    omp_set_num_threads((int)num_devices);
    //omp_set_num_threads(1);
   
    // Times for each
    cl_double* kernel_times=(cl_double*)calloc(num_devices, sizeof(cl_double));

    // Loop over experiments
    for (int n=0; n<nstats; n++) {
        
        // Loop over domains using dynamic scheduling
        #pragma omp parallel for shared(command_queues, As_d, Bs_d, Cs_d, C_h, D0, D1, N0_C, N1_C, L0, L1, nbytes_C, kernel_times) default(none) schedule(dynamic,1)  
        for (int d=0; d<D0*D1; d++) {
        
            // A is of size (m, k)
            // B is of size (k, n)
            // C is of size (m, n)
        
            // Local domain indices
            size_t l0 = d/D1;
            size_t l1 = d%D1;
            
            size_t start0 = l0*L0;
            size_t start1 = l1*L1;
            
            size_t stop0 = std::min((l0+1)*L0,(size_t)N0_C);
            size_t stop1 = std::min((l1+1)*L1,(size_t)N1_C);
        
            // Get the thread ID
            int tid = omp_get_thread_num();
        
            // size of the local domain
            size_t s0 = stop0-start0;
            size_t s1 = stop1-start1;
        
            // starting positions in the matrices
            size_t offset_A = start0*NCOLS_A;
            size_t offset_B = start1;
            size_t offset_C = start0*NCOLS_C+start1;
        
            // Start the clock
            auto t1 = std::chrono::high_resolution_clock::now();
            
            // Event for the kernel
            cl_event kernel_event;
        
            // Leading dimension is number of elements that forms the biggest stride
            CLBlastStatusCode status = CLBlastSgemm(
                CLBlastLayoutRowMajor,
                CLBlastTransposeNo,
                CLBlastTransposeNo,
                // Size of region in dim 0 of C
                (const size_t)s0,
                // Size of region in dim 1 of C
                (const size_t)s1,
                // Size of region in dim 1 of A
                (const size_t)NCOLS_A,
                alpha,
                As_d[tid], (const size_t)offset_A, (const size_t)NCOLS_A,
                Bs_d[tid], (const size_t)offset_B, (const size_t)NCOLS_C,
                beta,
                Cs_d[tid], (const size_t)offset_C, (const size_t)NCOLS_C,
                &command_queues[tid],
                &kernel_event
            );
        
            // Make sure the matrix multiplication ran successfully
            assert(status==CLBlastSuccess);
            
            // Wait for events to finish
            H_ERRCHK(
                clWaitForEvents(
                    1,
                    &kernel_event
                )
            );

            auto t2 = std::chrono::high_resolution_clock::now();
        
            // Record the cumulative kernel time in milliseconds
            kernel_times[tid] += (cl_double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;
            
           // Copy the domain back to C
            
            // B is of size (N1_A, N1_C)
            // Offset is in bytes, row_id and slice_id are indices
            size_t offset=start1*sizeof(cl_float), row_id=start0, slice_id = 0;
    
            // Make up the origin for host and buffer
            const size_t buffer_origin[] = {offset, row_id, slice_id};
            const size_t host_origin[] = {offset, row_id, slice_id};
    
            // Length of a row (in bytes)
            size_t buffer_row_pitch = NCOLS_C * sizeof(cl_float); 
            size_t host_row_pitch = buffer_row_pitch;
    
            // Number of bytes in a slice 
            size_t buffer_slice_pitch = NROWS_C * NCOLS_C * sizeof(cl_float);
            size_t host_slice_pitch = buffer_slice_pitch;        
        
            /// Size of the region to copy, of course we only copy 1 slice
            size_t nrows = s0, nslices = 1;
            const size_t region[] = { s1*sizeof(cl_float), nrows, nslices};
     
            cl_float zero=1.0;
            
            // Enqueue the rectangular copy
            H_ERRCHK(
                clEnqueueReadBufferRect(
                    command_queues[tid],
                    Cs_d[tid],
                    CL_TRUE,
                    buffer_origin,
                    host_origin,
                    region,
                    buffer_row_pitch,
                    buffer_slice_pitch,
                    host_row_pitch,
                    host_slice_pitch,
                    C_h,
                    0,
                    NULL,
                    NULL
                )
            );
        } // End of parallel region
        
        // Use the maximum cumulative time on a thread as the kernel time
        cl_double time_ms=0.0;
        for (cl_uint n=0; n<num_devices; n++) {
            time_ms = fmax(time_ms, kernel_times[n]);
            kernel_times[n]=0.0;
        } 

        // Keep track of maximum time
        if (time_ms > max_time_ms) {
            max_time_ms = time_ms;
            max_time_n = n;
        }
        
        // Fetch parallel times
        times_ms[n]=time_ms;
        avg_time_ms+=time_ms;
    }

    // Kernel times is no longer required
    free(kernel_times);

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

    for (cl_uint n=0; n<num_devices; n++) {
        // Free the OpenCL buffers
        H_ERRCHK(clReleaseMemObject(As_d[n]));
        H_ERRCHK(clReleaseMemObject(Bs_d[n]));
        H_ERRCHK(clReleaseMemObject(Cs_d[n]));
    }
 
    // Free the buffers arrays
    free(As_d);
    free(Bs_d);
    free(Cs_d);
    
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

