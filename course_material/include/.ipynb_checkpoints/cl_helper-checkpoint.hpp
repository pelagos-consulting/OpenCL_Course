#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cassert>
#include <cstring>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.hpp"
#endif

#define OCL_EXIT -20

// Make a lookup table for error codes
std::map<cl_int, const char*> error_codes {
    {CL_SUCCESS, "CL_SUCCESS"},
    {CL_BUILD_PROGRAM_FAILURE, "CL_BUILD_PROGRAM_FAILURE"},
    {CL_COMPILE_PROGRAM_FAILURE, "CL_COMPILE_PROGRAM_FAILURE"},
    {CL_COMPILER_NOT_AVAILABLE, "CL_COMPILER_NOT_AVAILABLE"},
    {CL_DEVICE_NOT_FOUND, "CL_DEVICE_NOT_FOUND"},
    {CL_DEVICE_NOT_AVAILABLE, "CL_DEVICE_NOT_AVAILABLE"},
    {CL_DEVICE_PARTITION_FAILED, "CL_DEVICE_PARTITION_FAILED"},
    {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {CL_IMAGE_FORMAT_MISMATCH, "CL_IMAGE_FORMAT_MISMATCH"},
    {CL_IMAGE_FORMAT_NOT_SUPPORTED, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {CL_INVALID_ARG_INDEX, "CL_INVALID_ARG_INDEX"},
    {CL_INVALID_ARG_SIZE, "CL_INVALID_ARG_SIZE"},
    {CL_INVALID_ARG_VALUE, "CL_INVALID_ARG_VALUE"},
    {CL_INVALID_BINARY, "CL_INVALID_BINARY"},
    {CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE"},
    {CL_INVALID_BUILD_OPTIONS, "CL_INVALID_BUILD_OPTIONS"},
    {CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE"},
    {CL_INVALID_COMPILER_OPTIONS, "CL_INVALID_COMPILER_OPTIONS"},
    {CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT"},
    {CL_INVALID_DEVICE, "CL_INVALID_DEVICE"},
    {CL_INVALID_DEVICE_PARTITION_COUNT, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {CL_INVALID_DEVICE_TYPE, "CL_INVALID_DEVICE_TYPE"},
    {CL_INVALID_EVENT, "CL_INVALID_EVENT"},
    {CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_WAIT_LIST"},
    {CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET"},
    {CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {CL_INVALID_HOST_PTR, "CL_INVALID_HOST_PTR"},
    {CL_INVALID_IMAGE_DESCRIPTOR, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {CL_INVALID_IMAGE_SIZE, "CL_INVALID_IMAGE_SIZE"},
    {CL_INVALID_KERNEL, "CL_INVALID_KERNEL"},
    {CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS"},
    {CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION"},
    {CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME"},
    {CL_INVALID_LINKER_OPTIONS, "CL_INVALID_LINKER_OPTIONS"},
    {CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT"},
    {CL_INVALID_OPERATION, "CL_INVALID_OPERATION"},
    {CL_INVALID_PLATFORM, "CL_INVALID_PLATFORM"},
    {CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM"},
    {CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {CL_INVALID_PROPERTY, "CL_INVALID_PROPERTY"},
    {CL_INVALID_QUEUE_PROPERTIES, "CL_INVALID_QUEUE_PROPERTIES"},
    {CL_INVALID_SAMPLER, "CL_INVALID_SAMPLER"},
    {CL_INVALID_VALUE, "CL_INVALID_VALUE"},
    {CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION"},
    {CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE"},
    {CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE"},
    {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {CL_LINK_PROGRAM_FAILURE, "CL_LINK_PROGRAM_FAILURE"},
    {CL_LINKER_NOT_AVAILABLE, "CL_LINKER_NOT_AVAILABLE"},
    {CL_MAP_FAILURE, "CL_MAP_FAILURE"},
    {CL_MEM_COPY_OVERLAP, "CL_MEM_COPY_OVERLAP"},
    {CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY"},
    {CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES"},
    {CL_PROFILING_INFO_NOT_AVAILABLE, "CL_PROFILING_INFO_NOT_AVAILABLE"},
};

// Function to check error code
void h_errchk(cl_int errcode, const char *message) {
    if (errcode!=CL_SUCCESS) {
        // Is the error code in the map
        if (error_codes.count(errcode)>0) {
            std::printf("Error, Opencl call failed at \"%s\" with error code %s (%d)\n", 
                    message, error_codes[errcode], errcode);
        } else {
            // We don't know how to handle the error code, so just print it
            std::printf("Error, OpenCL call failed at \"%s\" with error code %d\n", 
                    message, errcode);
        }
        // We have failed one way or the other, so just exit
        exit(OCL_EXIT);
    }
};

// Function to create lists of contexts and devices that map to available hardware
void h_acquire_devices(
        // Input parameter
        cl_device_type device_type,
        // Output parameters
        cl_platform_id **platform_ids_out,
        cl_uint *num_platforms_out,
        cl_device_id **device_ids_out,
        cl_uint *num_devices_out, 
        cl_context **contexts_out) {

    // Return code for running things
    cl_int errcode = CL_SUCCESS;
    
    //// Get all valid platforms ////
    cl_uint num_platforms; 
    cl_platform_id *platform_ids = NULL;
    
    // First call to clGetPlatformIDs - get the number of platforms
    h_errchk(clGetPlatformIDs(0, NULL, &num_platforms), "Fetching number of platforms");
    
    // Allocate memory for platform id's
    platform_ids = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    
    // Second call to clGetPlatformIDs - fill the platforms
    h_errchk(clGetPlatformIDs(num_platforms, platform_ids, NULL), "Fetching platforms");
        
    // Fetch the total number of compatible devices
    cl_uint num_devices=0;
    
    // Loop over each platform and get the total number
    // of devices that match device_type
    for (cl_uint n=0; n < num_platforms; n++) {
        // Temporary number of devices
        cl_uint ndevs;
        // Get number of devices in the platform
        errcode = clGetDeviceIDs(
            platform_ids[n],
            device_type,
            0,
            NULL,
            &ndevs);

        if (errcode != CL_DEVICE_NOT_FOUND) {
            h_errchk(errcode, "Getting number of devices");
            num_devices += ndevs;
        }
    }
    
    // Check to make sure we have more than one suitable device
    if (num_devices == 0) {
        std::printf("Failed to find a suitable compute device\n");
        exit(OCL_EXIT);
    }

    // Allocate flat 1D allocations for device ID's and contexts,
    // both allocations have the same number of elements
    cl_device_id *device_ids = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
    cl_context *contexts = (cl_context*)calloc(num_devices, sizeof(cl_context));
    
    // Temporary pointers
    cl_device_id *device_ids_ptr = device_ids;
    cl_context *contexts_ptr = contexts;
    
    // Now loop over platforms and fill device ID's array
    for (cl_uint n=0; n < num_platforms; n++) {
        // Temporary number of devices
        cl_uint ndevs;

        // Get the number of devices in a platform
        errcode = clGetDeviceIDs(
            platform_ids[n],
            device_type,
            0,
            NULL,
            &ndevs);

        if (errcode != CL_DEVICE_NOT_FOUND) {
            // Check to see if any other error was generated
            h_errchk(errcode, "Getting number of devices for the platform");
            
            // Fill the array with the next set of found devices
            h_errchk(clGetDeviceIDs(
                platform_ids[n],
                device_type,
                ndevs,
                device_ids_ptr,
                NULL), "Filling devices");
            
            // Create a context for every device found
            for (cl_uint c=0; c<ndevs; c++ ) {
                // Context properties, this can be tricky
                const cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, 
                                                      (cl_context_properties)platform_ids[n], 
                                                      0 };
                
                // Create a context with 1 device in it
                const cl_device_id dev_id = device_ids_ptr[c];
                cl_uint ndev = 1;
                
                // Fill the contexts array at this point 
                // with a newly created context
                *contexts_ptr = clCreateContext(
                    prop, 
                    ndev, 
                    &dev_id,
                    NULL,
                    NULL,
                    &errcode
                );
                h_errchk(errcode, "Creating a context");
                contexts_ptr++;
            }
            
            // Advance device_id's pointer 
            // by the number of devices discovered
            device_ids_ptr += ndevs;
        }
    }   

    // Fill in output information here to 
    // avoid problems with understanding
    *platform_ids_out = platform_ids;
    *num_platforms_out = num_platforms;
    *device_ids_out = device_ids;
    *num_devices_out = num_devices;
    *contexts_out = contexts;
}

// Function to create command queues
cl_command_queue* h_create_command_queues(
        // Create a list of command queues
        // with selectable properties
        // Assumes that contexts is as long as devices
        
        // Array of OpenCL device id's
        cl_device_id *devices,
        // Array of OpenCL contexts
        cl_context *contexts,
        // How long is devices and contexts?
        cl_uint num_devices,
        // How many command queues should we create?
        cl_uint num_command_queues,
        // Do we enable out-of-order execution?
        cl_bool out_of_order_enable,
        // Do we enable profiling of commands 
        // sent to the command queues
        cl_bool profiling_enable) {
    
    // Return code for error checking
    cl_int errcode;   

    // Manage bit fields for the command queue properties
    cl_command_queue_properties queue_properties = 0;
    if (out_of_order_enable == CL_TRUE) {
        queue_properties = queue_properties | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;    
    }
    if (profiling_enable == CL_TRUE) {
        queue_properties = queue_properties | CL_QUEUE_PROFILING_ENABLE;    
    }

    // Allocate memory for the command queues
    cl_command_queue *command_queues = (cl_command_queue*)calloc(num_command_queues, sizeof(cl_command_queue));

    // Fill command queues in a Round-Robin fashion
    for (cl_uint n=0; n<num_command_queues; n++) {
        command_queues[n] = clCreateCommandQueue(
            contexts[n % num_devices],
            devices[n % num_devices],
            queue_properties,
            &errcode    
        );
        h_errchk(errcode, "Creating a command queue");        
    }
            
    return command_queues;
}


// Function to build a program from a single device and context
cl_program h_build_program(const char* source, 
                           cl_context context, 
                           cl_device_id device,
                           const char* compiler_options) {

    // Error code for checking programs
    cl_int errcode;

    // Create a program from the source code
    cl_program program = clCreateProgramWithSource(
            context,
            1,
            (const char**)(&source),
            NULL,
            &errcode
    );
    h_errchk(errcode, "Creating OpenCL program");

    // Try to compile the program, print a log otherwise
    errcode = clBuildProgram(program, 
                1, 
                &device,
                compiler_options,
                NULL,
                NULL
    );

    // If the compilation process failed then fetch a build log
    if (errcode!=CL_SUCCESS) {
        // Number of characters in the build log
        size_t nchars;
        
        // Query the size of the build log
        h_errchk(clGetProgramBuildInfo( program,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        0,
                                        NULL,
                                        &nchars),"Checking build log");

        // Make up the build log string
        char* buildlog=(char*)calloc(nchars+1, sizeof(char));

        // Query the build log 
        h_errchk(clGetProgramBuildInfo( program,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        nchars,
                                        buildlog,
                                        NULL), "Filling the build log");
        
        // Insert a NULL character at the end of the string
        buildlog[nchars] = '\0';
        
        // Print the build log
        std::printf("Build log is %s\n", buildlog);
        free(buildlog);
        exit(OCL_EXIT);
    }

    return program;
}

cl_double h_get_io_rate_MBs(cl_double time_ms, size_t nbytes) {
    // Get the IO rate in MB/s for bytes read or written
    return (cl_double)nbytes * 1.0e-3 / time_ms;
}

cl_double h_get_event_time_ms(
        cl_event *event, 
        const char* message, 
        size_t* nbytes) {
    
    // Make sure the event has finished
    h_errchk(clWaitForEvents(1, event), message);
    
    // Start and end times
    cl_ulong t1, t2;
        
    // Fetch the start and end times in nanoseconds
    h_errchk(
        clGetEventProfilingInfo(
            *event,
            CL_PROFILING_COMMAND_START,
            sizeof(cl_ulong),
            &t1,
            NULL
        ),
        "Fetching start time for event"
    );

    h_errchk(
        clGetEventProfilingInfo(
            *event,
            CL_PROFILING_COMMAND_END,
            sizeof(cl_ulong),
            &t2,
            NULL
        ),
        "Fetching end time for event"
    );
    
    // Convert the time into milliseconds
    cl_double elapsed = (cl_double)(t2-t1)*(cl_double)1.0e-6;
        
    // Print the timing message if necessary
    if (strlen(message)>0) {
        std::printf("Time for event \"%s\": %.3f ms", message, elapsed);
        
        // Print transfer rate if nbytes is not NULL
        if (nbytes != NULL) {
            cl_double io_rate_MBs = h_get_io_rate_MBs(
                elapsed, 
                *nbytes
            );
            std::printf(" (%.2f MB/s)", io_rate_MBs);
        }
        std::printf("\n");
    }
    
    return elapsed;
}

void h_fit_global_size(const size_t* global_size, const size_t* local_size, size_t work_dim) {
    // Fit global size so that an integer number of local sizes fits within it in any dimension
    
    // Make a readable pointer out of the constant one
    size_t* new_global = (size_t*)global_size;
    
    // Make sure global size is large enough
    for (int n=0; n<work_dim; n++) {
        assert(global_size[n]>0);
        assert(global_size[n]>=local_size[n]);
        if ((global_size[n] % local_size[n]) > 0) {
            new_global[n] = ((global_size[n]/local_size[n])+1)*local_size[n];
        } 
    }
}

void h_write_binary(void* data, const char* filename, size_t nbytes) {
    // Write binary data to file
    std::FILE *fp = std::fopen(filename, "wb");
    if (fp == NULL) {
        std::printf("Error in writing file %s", filename);
        exit(OCL_EXIT);
    }
    
    // Write the data to file
    std::fwrite(data, nbytes, 1, fp);
    
    // Close the file
    std::fclose(fp);
}

void* h_read_binary(const char* filename, size_t *nbytes) {
    // Open the file for reading and use std::fread to read in the file
    std::FILE *fp = std::fopen(filename, "rb");
    if (fp == NULL) {
        std::printf("Error in reading file %s", filename);
        exit(OCL_EXIT);
    }
    
    // Seek to the end of the file
    std::fseek(fp, 0, SEEK_END);
    
    // Extract the number of bytes in this file
    *nbytes = std::ftell(fp);

    // Rewind the file pointer
    std::rewind(fp);

    // Create a buffer to read into
    // Add an extra Byte for a null termination character
    // just in case we are reading to a string
    void *buffer = calloc((*nbytes)+1, 1);
    
    // Set the NULL termination character
    char* source = (char*)buffer;
    source[*nbytes] = '\0';
    
    // Read the file into the buffer and close
    std::fread(buffer, 1, *nbytes, fp);
    std::fclose(fp);
    return buffer;
}

// Function to report information on a compute device
void h_report_on_device(cl_device_id device) {
    // Report some information on the device
    
    // Fetch the name of the compute device
    size_t nbytes_name;
    
    // First call is to fetch 
    // the number of bytes taken up by the name
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_NAME, 
                        0, 
                        NULL, 
                        &nbytes_name),
        "Device name bytes"
    );
    // Allocate memory for the name
    char* name=new char[nbytes_name+1];
    // Don't forget the NULL character terminator
    name[nbytes_name] = '\0';
    // Second call is to fill the allocated name
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_NAME, 
                        nbytes_name, 
                        name, 
                        NULL),
        "Device name"
    );
    std::printf("\t%20s %s \n","name:", name);

    // Fetch the global memory size
    cl_ulong mem_size;
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_GLOBAL_MEM_SIZE, 
                        sizeof(cl_ulong), 
                        &mem_size, 
                        NULL),
        "Global mem size"
    );
    std::printf("\t%20s %llu MB\n","global memory size:",mem_size/(1000000));
    
    // Fetch the maximum size of a global memory allocation
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_MAX_MEM_ALLOC_SIZE, 
                        sizeof(cl_ulong), 
                        &mem_size, 
                        NULL),
        "Max mem alloc size"
    );
    std::printf("\t%20s %llu MB\n","max buffer size:", mem_size/(1000000));
    
    // Get the maximum number of dimensions supported
    cl_uint max_work_dims;
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 
                        sizeof(cl_uint), 
                        &max_work_dims, 
                        NULL),
        "Max number of dimensions for local size."
    );
    
    // Get the max number of work items along
    // dimensions of a work group
    size_t* max_size = new size_t[max_work_dims];
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_MAX_WORK_ITEM_SIZES, 
                        max_work_dims*sizeof(size_t), 
                        max_size, 
                        NULL),
        "Max size for work items."
    );
    
    // Print out the maximum extent of 
    // items in a workgroup
    std::printf("\t%20s (", "max local size:");
    for (int n=0; n<max_work_dims-1; n++) {
        std::printf("%zu,", max_size[n]);
    }
    std::printf("%zu)\n", max_size[max_work_dims-1]);
    
    // Get the maximum number of work items in a work group
    size_t max_work_group_size;
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                        sizeof(size_t), 
                        &max_work_group_size, 
                        NULL),
        "Max number of work-items a workgroup."
    );
    std::printf("\t%20s %zu\n", "max work-items:", max_work_group_size);
    
    // Clean up
    delete [] max_size;
    delete [] name;
}

// Function to release command queues
void h_release_command_queues(cl_command_queue *command_queues, cl_uint num_command_queues) {
    // Finish and Release all command queues
    for (cl_uint n = 0; n<num_command_queues; n++) {
        // Wait for all commands in the 
        // command queues to finish
        h_errchk(
            clFinish(command_queues[n]), 
            "Finishing up command queues"
        );
        
        // Now release the command queue
        h_errchk(
            clReleaseCommandQueue(command_queues[n]), 
            "Releasing command queues"
        );
    }

    // Now destroy the command queues
    free(command_queues);
}

// Function to release devices and contexts
void h_release_devices(
        cl_device_id *devices,
        cl_uint num_devices,
        cl_context* contexts,
        cl_platform_id *platforms) {
    
    // Release contexts and devices
    for (cl_uint n = 0; n<num_devices; n++) {
        h_errchk(
            clReleaseContext(contexts[n]), 
            "Releasing context"
        );

        h_errchk(
            clReleaseDevice(devices[n]), 
            "Releasing device"
        );
    }

    // Free all arrays allocated with h_acquire_devices
    free(contexts);
    free(devices);
    free(platforms);
}
