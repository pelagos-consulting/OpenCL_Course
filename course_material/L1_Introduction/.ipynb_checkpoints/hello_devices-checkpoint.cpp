// Include the OpenCL helper headers
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cassert>
#include <cstring>
#include <cmath>
#include <chrono>

/// Define target OpenCL version
#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/opencl.h"
#endif

/// Lookup table for error codes
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

/// Check error codes
void h_errchk(cl_int errcode, const char *message) {
    if (errcode!=CL_SUCCESS) {
        // Is the error code in the map
        if (error_codes.count(errcode)>0) {
            std::fprintf(stderr, "Error, Opencl call failed at \"%s\" with error code %s (%d)\n", 
                    message, error_codes[errcode], errcode);
        } else {
            // We don't know how to handle the error code, so just print it
            std::fprintf(stderr, "Error, OpenCL call failed at \"%s\" with error code %d\n", 
                    message, errcode);
        }
        // We have failed one way or the other, so just exit
        exit(EXIT_FAILURE);
    }
};

/// Macro to check error codes, but with file and line information
#define H_ERRCHK(cmd) \
{\
    std::string msg = __FILE__;\
    msg += ":" + std::to_string(__LINE__);\
    h_errchk(cmd, msg.c_str());\
}

// Main program
int main(int argc, char** argv) {
    
    // Error code for checking the status of 
    cl_int errcode = CL_SUCCESS;    
    
    // Choose which kind of device we want to use
    // Could also be 
    // CL_DEVICE_TYPE_GPU
    // CL_DEVICE_TYPE_CPU
    // CL_DEVICE_TYPE_ACCELERATOR
    cl_device_type device_type = CL_DEVICE_TYPE_ALL;
    
    //// Get all valid platforms ////
    cl_uint num_platforms; 
    cl_platform_id *platform_ids = NULL;
    // First call to clGetPlatformIDs - get the number of platforms
    H_ERRCHK(clGetPlatformIDs(0, NULL, &num_platforms));
    // Allocate memory for platform ID's
    platform_ids = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    // Second call to clGetPlatformIDs - fill the platforms
    H_ERRCHK(clGetPlatformIDs(num_platforms, platform_ids, NULL));
    
    // Device index
    cl_uint dev_index=0;
    
    // Loop over the number of platforms
    // and query devices in each platform
    for (cl_uint n=0; n < num_platforms; n++) {
        
        std::printf("Platform %d\n", n);
        
        // Temporary number of devices
        cl_uint ndevs;
        
        // First call to clGetDeviceIDs - Get number of devices in the platform
        errcode = clGetDeviceIDs(
            platform_ids[n],
            device_type,
            0,
            NULL,
            &ndevs);

        if (errcode != CL_DEVICE_NOT_FOUND) {
            // Then there are devices on the platform and we can proceed
            H_ERRCHK(errcode);
 
            // Allocate memory for device ID's
            cl_device_id *device_ids = (cl_device_id*)calloc(ndevs, sizeof(cl_device_id));
            
            // Fill device ID's
            H_ERRCHK(clGetDeviceIDs(
                platform_ids[n],
                device_type,
                ndevs,
                device_ids,
                NULL)
            );      
            
            // Loop over every device in a platform and make enquiries
            for (cl_uint i=0; i<ndevs; i++) {
                
                // Print the device index
                std::printf("\t%22s %d\n", "Device index:", dev_index++);                             
                
                // Fetch the name of the compute device
                size_t nbytes_name;
                // Fetch the number of bytes taken up by the name
                H_ERRCHK(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, 0, NULL, &nbytes_name));
                // Allocate memory for the name
                char* name=new char[nbytes_name+1];
                // Don't forget the NULL character terminator
                name[nbytes_name] = '\0';
                // Second call is to fill the allocated name
                H_ERRCHK(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, nbytes_name, name, NULL));
                std::printf("\t%22s %s \n","name:", name);
                delete [] name;
                
                // Fetch the global memory size
                cl_ulong mem_size;
                H_ERRCHK(clGetDeviceInfo(device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE, 
                                         sizeof(cl_ulong), &mem_size, NULL));
                std::printf("\t%22s %lu MB\n","global memory size:",mem_size/(1000000));
                
            }
            
            // Release every device in the platform
            for (cl_uint i=0; i<ndevs; i++) {
                H_ERRCHK(clReleaseDevice(device_ids[i]));
            }
            
            // Free memory for devices
            free(device_ids);
            
        }
    }
    
    // Release the array for platform ID's
    free(platform_ids);
}