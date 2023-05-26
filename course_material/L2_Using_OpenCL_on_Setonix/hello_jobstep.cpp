/**********************************************************
"Hello World"-type program to test different srun layouts.

Written by Tom Papatheodore, modified for OpenCL by Toby Potter
**********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <iomanip>
#include <string.h>
#include <mpi.h>
#include <sched.h>
#include <omp.h>
#include <map>
#include <sstream>

// Define target OpenCL version
#define CL_TARGET_OPENCL_VERSION 120

#include "CL/opencl.h"
#include "CL/cl_ext.h"

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

// Function to check error codes
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

// Macro to check error codes
#define H_ERRCHK(cmd) \
{\
    std::string msg = __FILE__;\
    msg += ":" + std::to_string(__LINE__);\
    h_errchk(cmd, msg.c_str());\
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
        exit(EXIT_FAILURE);
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

// Structure to hold AMD topology information
typedef union
 {
     struct { cl_uint type; cl_uint data[5]; } raw;
     struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
 } cl_device_topology_amd;

int main(int argc, char *argv[]){

    // Initialise MPI
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char name[MPI_MAX_PROCESSOR_NAME];
	int resultlength;
	MPI_Get_processor_name(name, &resultlength);

    // If ROCR_VISIBLE_DEVICES is set, capture visible GPUs
    const char* gpu_id_list; 
    const char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
    if(rocr_visible_devices == NULL){
        gpu_id_list = "N/A";
    }
    else{
        gpu_id_list = rocr_visible_devices;
    }

    // Errorcode
    cl_int errcode = CL_SUCCESS;
    
    // Set the device type
    cl_device_type target_device = CL_DEVICE_TYPE_GPU;
    
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

    // Hardware thread
	unsigned int hwthread;
    // OpenMP thread
	int thread_id = 0;
	unsigned int numa_id = 0;

	if(num_devices == 0){
		#pragma omp parallel default(shared) private(hwthread, thread_id, numa_id)
		{
			thread_id = omp_get_thread_num();
			getcpu(&hwthread, &numa_id);

            printf("MPI %03d - OMP %03d - HWT %03d - NUMA %03d - Node %s\n", 
                    rank, thread_id, hwthread, numa_id, name);

		}
	} else {

		char busid[64];

        std::string busid_list = "";
        std::string rt_gpu_id_list = "";

		// Loop over the GPUs available to each MPI rank
		for(int i=0; i<num_devices; i++){

            std::stringstream temp;

#ifdef CL_DEVICE_TOPOLOGY_AMD	    
            // Bus ID query for AMD devices
            cl_device_topology_amd top;
            errcode = clGetDeviceInfo(devices[i], 
                CL_DEVICE_TOPOLOGY_AMD,
                sizeof(cl_device_topology_amd),
                &top,
                NULL
            );
            if (errcode==CL_SUCCESS) {
                // Convert the bus ID to hex
                temp << std::hex << (int)top.pcie.bus;
            }
#endif

#ifdef CL_DEVICE_PCI_BUS_ID_NV
            // Bus ID query for NVIDIA devices
            cl_int nv_id;
            errcode = clGetDeviceInfo(devices[i], 
                CL_DEVICE_PCI_SLOT_ID_NV,
                sizeof(cl_int),
                &nv_id,
                NULL
            );
            if (errcode==CL_SUCCESS) {
                // Convert the bus ID to hex
                temp << std::hex << nv_id;
            }
#endif
	    // Concatenate per-MPIrank GPU info into strings for print
            if(i > 0) rt_gpu_id_list.append(",");
            rt_gpu_id_list.append(std::to_string(i));

            if(i > 0) busid_list.append(",");
            busid_list.append(temp.str().substr(6,2));

		}

		#pragma omp parallel default(shared) private(hwthread, numa_id, thread_id)
		{
            #pragma omp critical
            {
			thread_id = omp_get_thread_num();
			getcpu(&hwthread, &numa_id);

            printf("MPI %03d - OMP %03d - HWT %03d - NUMA_ID %03d - Node %s - RT_GPU_ID %s - GPU_ID %s - Bus_ID %s\n",
                    rank, thread_id, hwthread, numa_id, name, rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str());
           }
		}
	}

    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms
    );
    
    // Clean up MPI
	MPI_Finalize();

	return 0;
}

