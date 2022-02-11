#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.hpp"
#endif

#define OCL_EXIT -20

// Make a lookup table for error codes
std::map<cl_int, std::string> error_codes {
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
void h_errchk(cl_int errcode, std::string message) {
    if (errcode!=CL_SUCCESS) {
        if (error_codes.count(errcode)>0) {
            std::printf("Error, Opencl call failed at \"%s\" with error code %s (%d)\n", 
                    message.c_str(), error_codes[errcode].c_str(), errcode);
        } else {
            std::printf("Error, OpenCL call failed at \"%s\" with error code %d\n", 
                    message.c_str(), errcode);
        }
        exit(OCL_EXIT);
    }
};

// Function to create command queues
cl_command_queue* h_create_command_queues(
        // Create a list of command queues
        // with selectable properties
        // Assumes that contexts is as long as devices
        cl_device_id *devices,
        cl_context *contexts, 
        cl_uint num_devices, 
        cl_uint num_command_queues,
        cl_bool out_of_order_enable,
        cl_bool profiling_enable) {
    
    cl_int ret_code;   

    // Manage bit fields
    cl_command_queue_properties queue_properties = 0;
    if (out_of_order_enable == CL_TRUE) {
        queue_properties = queue_properties | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;    
    }
    if (profiling_enable == CL_TRUE) {
        queue_properties = queue_properties | CL_QUEUE_PROFILING_ENABLE;    
    }

    cl_command_queue *command_queues = (cl_command_queue*)calloc(num_command_queues, sizeof(cl_command_queue));

    // Allocate command queues in a Round-Robin fashion
    for (cl_uint n=0; n<num_command_queues; n++) {
        command_queues[n] = clCreateCommandQueue(
            contexts[n % num_devices],
            devices[n % num_devices],
            queue_properties,
            &ret_code    
        );
        h_errchk(ret_code, "Creating a context");        
    }
            
    return command_queues;
}

// Function to release command queues
void h_release_command_queues(cl_command_queue *command_queues, cl_uint num_command_queues) {
    // Release command queues
    for (cl_uint n = 0; n<num_command_queues; n++) {
        h_errchk(clFinish(command_queues[n]), "Finishing up command queues");
        h_errchk(clReleaseCommandQueue(command_queues[n]), "Releasing command queues");
    }

    // Now destroy the command queues
    free(command_queues);
}

// Function to build a program from a single device and context
cl_program h_build_program(const char* source, cl_context context, cl_device_id device) {

    cl_int ret_code;

    cl_program program = clCreateProgramWithSource(
            context,
            1,
            (const char**)(&source),
            NULL,
            &ret_code);
        h_errchk(ret_code, "Creating OpenCL program");

    // Try to build the program, print a log otherwise
    ret_code = clBuildProgram(program, 
                1, 
                &device,
                NULL,
                NULL,
                NULL);

    if (ret_code!=CL_SUCCESS) {
        size_t elements;
        h_errchk(clGetProgramBuildInfo( program,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        0,
                                        NULL,
                                        &elements),"Checking build log");

        // Make up the build log string
        char* buildlog=(char*)calloc(elements, 1);

        h_errchk(clGetProgramBuildInfo( program,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        elements,
                                        buildlog,
                                        NULL), "Filling the build log");
        std::printf("Build log is %s\n", buildlog);
        free(buildlog);
        exit(OCL_EXIT);
    }

    return program;
}

void h_fit_global_size(const size_t* global_size, const size_t* local_size, size_t work_dim) {
    // Fit global size so that an integer number of local sizes fits within it in any dimension
    
    // Make a readable pointer out of the constant one
    size_t* new_global = (size_t*)global_size;
    
    // Make sure global size is large enough
    for (int n=0; n<work_dim; n++) {
        assert(global_size[n]>0);
        assert(global_size[n]>=local_size[n]);
        if (global_size[n] % local_size[n]) {
            new_global[n] = ((global_size[n]/local_size[n])+1)*local_size[n];
        } 
    }
}

void h_write_binary(void* data, const char* filename, size_t nbytes) {
    // Write binary data to file
    std::FILE *fp = std::fopen(filename, "wb");
    if (fp == NULL) {
        std::printf("Error in writing OpenCL source file %s", filename);
        exit(OCL_EXIT);
    }
    
    // Write the data to file
    std::fwrite(data, nbytes, 1, fp);
    
    // Close the file
    std::fclose(fp);
}

void* h_read_binary(const char* filename, size_t *nbytes) {
    // Open the file for reading and use std::fread to read in the file
    // Add a termination character at the end just in case we are reading to a string
    std::FILE *fp = std::fopen(filename, "rb");
    if (fp == NULL) {
        std::printf("Error in reading OpenCL source file %s", filename);
        exit(OCL_EXIT);
    }
    
    // Seek to the end of the file
    std::fseek(fp, 0, SEEK_END);
    
    // Extract the number of bytes in this file
    *nbytes = std::ftell(fp);

    // Rewind the file
    rewind(fp);

    // Create the Buffer to read into
    void *buffer = calloc((*nbytes)+1, 1);
    
    // Null Termination, in case this gets converted to string
    char* source = (char*)buffer;
    source[*nbytes] = '\0';
    std::fread(buffer, 1, *nbytes, fp);
    std::fclose(fp);
    return buffer;
}

// Function to report information on a compute device
void h_report_on_device(cl_device_id device) {

    // Report some information on the device
    
    // The name
    size_t nbytes_name;
    h_errchk(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &nbytes_name),"Device name bytes");
    char* name=new char[nbytes_name+1];
    name[nbytes_name] = '\0';
    h_errchk(clGetDeviceInfo(device, CL_DEVICE_NAME, nbytes_name, name, NULL),"Device name");
    int textwidth=16;
    std::printf("\t%20s %s \n","name:", name);

    // The global memory and max buffer sizes
    cl_ulong mem_size;
    h_errchk(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size,
    NULL),"Global mem size");
    std::printf("\t%20s %llu MB\n","global memory size:",mem_size/(1000000));
    
    h_errchk(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size,
    NULL),"Max mem alloc size");
    std::printf("\t%20s %llu MB\n","max buffer size:", mem_size/(1000000));
    
    // Get the maximum number of work items in a work group
    cl_uint max_work_dims;
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 
                        sizeof(cl_uint), 
                        &max_work_dims, 
                        NULL),
        "Max number of dimensions for local size."
    );
    
    // Get the maximum number of work items in a work group
    size_t* max_size = new size_t[max_work_dims];
    h_errchk(
        clGetDeviceInfo(device, 
                        CL_DEVICE_MAX_WORK_ITEM_SIZES, 
                        max_work_dims*sizeof(size_t), 
                        max_size, 
                        NULL),
        "Max size for work items."
    );
    
    // Print it out
    std::printf("\t%20s (", "max local size:");
    for (int n=0; n<max_work_dims-1; n++) {
        std::printf("%zu,", max_size[n]);
    }
    std::printf("%zu)\n", max_size[max_work_dims-1]);
    
    delete [] max_size;
    delete [] name;
}

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
    cl_int ret_code = CL_SUCCESS;
    
    // Get all valid platforms
    cl_uint num_platforms;
    cl_platform_id *platform_ids = NULL;
    h_errchk(clGetPlatformIDs(0, NULL, &num_platforms), "Fetching number of platforms");
    platform_ids = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    h_errchk(clGetPlatformIDs(num_platforms, platform_ids, NULL), "Fetching platforms");
        
    // Fetch the total number of compatible devices
    cl_uint num_devices=0;
    
    // Get the total number of devices
    for (cl_uint n=0; n < num_platforms; n++) {

        cl_uint ndevices;
        ret_code = clGetDeviceIDs(
            platform_ids[n],
            device_type,
            0,
            NULL,
            &ndevices);

        if (ret_code != CL_DEVICE_NOT_FOUND) {
            h_errchk(ret_code, "Getting number of devices");
            num_devices += ndevices;
        }
    }
    
    if (num_devices == 0) {
        std::printf("Failed to find a suitable compute device\n");
        exit(OCL_EXIT);
    }

    // Allocate memory for device ID's and contexts
    cl_device_id *device_ids = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
    cl_context *contexts = (cl_context*)calloc(num_devices, sizeof(cl_context));
    
    cl_device_id *device_ids_ptr = device_ids;
    cl_context *contexts_ptr = contexts;
    
    // Fill device ID's array
    for (cl_uint n=0; n < num_platforms; n++) {
        cl_uint ndevices;

        ret_code = clGetDeviceIDs(
            platform_ids[n],
            device_type,
            0,
            NULL,
            &ndevices);

        if (ret_code != CL_DEVICE_NOT_FOUND) {
            h_errchk(ret_code, "Getting number of devices for the platform");
            
            // Fill devices
            h_errchk(clGetDeviceIDs(
                platform_ids[n],
                device_type,
                ndevices,
                device_ids_ptr,
                NULL), "Filling devices");
            
            // Create a context for every device found
            for (cl_uint c=0; c<ndevices; c++ ) {
                // Context properties
                const cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, 
                                                      (cl_context_properties)platform_ids[n], 
                                                      0 };
                
                // Create a context with 1 device in it
                const cl_device_id dev_id = *(device_ids_ptr+c);
                cl_uint ndev = 1;
                
                *contexts_ptr = clCreateContext(
                    prop, 
                    ndev, 
                    &dev_id,
                    NULL,
                    NULL,
                    &ret_code
                );
                h_errchk(ret_code, "Creating a context");
                contexts_ptr++;
            }
            
            // Advance device_id's pointer
            device_ids_ptr += ndevices;
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

// Function to release devices and contexts
void h_release_devices(
        cl_device_id *devices,
        cl_uint num_devices,
        cl_context* contexts,
        cl_platform_id *platforms) {
    
    // Free contexts
    for (cl_uint n = 0; n<num_devices; n++) {
        h_errchk(clReleaseContext(contexts[n]), "Releasing contexts");
    }

    free(contexts);
    free(devices);
    free(platforms);
}