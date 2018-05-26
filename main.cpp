#include <iostream>
#include <stdexcept>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 220
#include "CL/cl.h"

using namespace std;

void checkStatus(cl_int status, string message) {
    if (status != CL_SUCCESS) {
        throw runtime_error(message + ". Status: " + to_string(status));
    }
}

void printPlatformInfo(const vector<cl_platform_id>& platformIds) {
    vector<char> stringBuffer (32);
    size_t stringSize;

    auto platformInfo = [&](cl_platform_id id, cl_platform_info infoEnum, string name) {
        cl_int status = clGetPlatformInfo(id, infoEnum, stringBuffer.size(), stringBuffer.data(), &stringSize);
        if (status == CL_SUCCESS) {
            cout << name << ": " << stringBuffer.data() << endl;
        }
    };

    for (cl_platform_id id : platformIds) {
        cout << "Platform Id: " << id << endl;
        platformInfo(id, CL_PLATFORM_NAME, "Name");
        platformInfo(id, CL_PLATFORM_PROFILE, "Profile");
        platformInfo(id, CL_PLATFORM_VERSION, "Version");
        platformInfo(id, CL_PLATFORM_VENDOR, "Vendor");
        platformInfo(id, CL_PLATFORM_EXTENSIONS, "Extensions");
    }

    cout << endl;
}

void printDeviceInfo(const vector<cl_device_id >& deviceIds) {
    vector<char> stringBuffer (32);
    size_t buffSizeWritten;

    auto platformInfo = [&](cl_device_id id, cl_device_info infoEnum, string name) {
        cl_int status = clGetDeviceInfo(id, infoEnum, stringBuffer.size(), stringBuffer.data(), &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << name << ": " << stringBuffer.data() << endl;
        }
    };

    auto boolToString = [](bool a) -> string { return a ? "Yes" : "No"; };

    for (cl_device_id id : deviceIds) {
        cout << "Device Id: " << id << endl;
        platformInfo(id, CL_DEVICE_NAME, "Name");
        cl_device_type type;
        cl_int status = clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(type), &type, &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << "Device Type: " <<  type << ". "
                 << "Is CPU: " << boolToString(type == CL_DEVICE_TYPE_CPU) << " "
                 << "Is GPU: " << boolToString(type == CL_DEVICE_TYPE_GPU) << endl;
        }

        cl_uint number;

        status = clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(number), &number, &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << "Device Max Compute Units: " << number << endl;
        }

        size_t sizeNum;
        status = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sizeNum), &sizeNum, &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << "Device Max Work Group Size: " << sizeNum << endl;
        }

        status = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(number), &number, &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << "Device Max Work Item Dimensions: " << number << endl;
        }

        size_t sizes[number];
        status = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), &sizes, &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << "Device Max Work Item Sizes: (";
            for (size_t i = 0; i < number; ++i) {
                cout << sizes[i] << ", ";
            }
            cout << ")" << endl;
        }

        status = clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(number), &number, &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << "Device Max Clock Frequency: " << number << " MHz" << endl;
        }

        cl_ulong memSize;
        status = clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, &buffSizeWritten);
        if (status == CL_SUCCESS) {
            cout << "Global Memory Size: " << memSize / (1024*1024) << " MB" << endl;
        }
    }

    cout << endl;
}

size_t getUserNumericInput(string message) {
    size_t result;
    cout << message << endl;
    cin >> result;
    return result;
}

//OpenCL kernel which is run for every work item created.
const char *saxpy_kernel =
        "__kernel                                   \n"
        "void saxpy_kernel(float alpha,             \n"
        "                  __global float *A,       \n"
        "                  __global float *B,       \n"
        "                  __global float *C        \n"
//        "                  uint maxIterations       \n"
        "                                    )      \n"
        "{                                          \n"
        "    //Get the index of the work-item       \n"
        "    int index = get_global_id(0);          \n"
//        "    uint i;                                 \n"
//        "    for(i = 0; i < maxIterations; ++i) {   \n"
        "        C[index] = alpha* A[index] + B[index]; \n"
//        "    }                                      \n"
        "}                                          \n";

int main() {

    try {
        // Generic return value
        cl_int status;

        // Query the number of platforms available
        cl_uint num_platforms;
        status = clGetPlatformIDs(0, nullptr, &num_platforms);
        checkStatus(status, "Cannot query the number of platforms available on this system");

        // Check whether there are supported devices in the first place
        if (num_platforms == 0) {
            cout << "No supported OpenCL platforms found" << endl;
            return EXIT_SUCCESS;
        }

        // Retrieve the id's of these platforms
        vector<cl_platform_id> platformIds(num_platforms);
        status = clGetPlatformIDs(num_platforms, platformIds.data(), nullptr);
        checkStatus(status, "Cannot query the identifiers of the platforms available on this system");
        printPlatformInfo(platformIds);

        // Query all devices available
        size_t platformNumber = platformIds.size() > 1 ? getUserNumericInput("Which platform would you like to use?") : 0;
        cl_platform_id platformId = platformIds.at(platformNumber);
        cout << "Selected platform id: " << platformId << endl << endl;

        // Get the device list in this platform
        cl_uint num_devices;
        status = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        checkStatus(status, "Cannot query the devices inside the selected platform");

        // Get the device Ids
        vector<cl_device_id> deviceIds (num_devices);
        status = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, num_devices, deviceIds.data(), nullptr);
        printDeviceInfo(deviceIds);

        // Select device to use
        size_t deviceNumber = deviceIds.size() > 1 ? getUserNumericInput("Which device would you like to use?") : 0;
        cl_device_id deviceId = deviceIds.at(deviceNumber);
        cout << "Selected device id: " << deviceId << endl << endl;

        // Create context - takes a list of devices but I'm only interested in one
        cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &status);
        checkStatus(status, "Cannot create context");

        // Create a command queue to send commands to the device
        // A command queue maps only with one device
        cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, &status);
        checkStatus(status, "Cannot create command queue");

        // Allocate data on host memory
        const size_t vecSize = 268435456 / 1024;
//        const uint32_t maxIterations = 1024;
        vector<float> a (vecSize, 1.f);
        vector<float> b (vecSize, 2.f);
        vector<float> c (vecSize, 0.f);

        // Create buffer on device memory with the same size as the host buffer
        cl_mem devBuffA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * a.size(), nullptr, &status);
        checkStatus(status, "Cannot allocate memory for vector A");
        cl_mem devBuffB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * b.size(), nullptr, &status);
        checkStatus(status, "Cannot allocate memory for vector B");
        cl_mem devBuffC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * c.size(), nullptr, &status);
        checkStatus(status, "Cannot allocate memory for vector C");

        // Enqueue a copy command to copy from host to device memory
        status = clEnqueueWriteBuffer(commandQueue, devBuffA, CL_TRUE, 0, sizeof(float) * a.size(), a.data(), 0, nullptr, nullptr);
        checkStatus(status, "Cannot copy memory for vector A");
        status = clEnqueueWriteBuffer(commandQueue, devBuffB, CL_TRUE, 0, sizeof(float) * b.size(), b.data(), 0, nullptr, nullptr);
        checkStatus(status, "Cannot copy memory for vector B");

        // Create a program
        cl_program program = clCreateProgramWithSource(context, 1, &saxpy_kernel, nullptr, &status);
        checkStatus(status, "Cannot create program with source");

        // Build it
        status = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
        checkStatus(status, "Cannot build program");

        // Create/Get the kernel from our program
        cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &status);
        checkStatus(status, "Cannot create kernel from program");

        // Set the arguments of the kernel
        const float alpha = 25.f;
        status = clSetKernelArg(kernel, 0, sizeof(float), &alpha);
        checkStatus(status, "Cannot set argument 0 of kernel");
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devBuffA);
        checkStatus(status, "Cannot set argument 1 of kernel");
        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &devBuffB);
        checkStatus(status, "Cannot set argument 2 of kernel");
        status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &devBuffC);
        checkStatus(status, "Cannot set argument 3 of kernel");
//        status = clSetKernelArg(kernel, 4, sizeof(uint32_t), &maxIterations);
//        checkStatus(status, "Cannot set argument 4 of kernel");

        // Execute the kernel :D
        size_t globalSizeOfItems = c.size();
        size_t blockSize = 128;
        status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &globalSizeOfItems, &blockSize, 0,
                                        nullptr, nullptr);
        checkStatus(status, "Could not execute the kernel");

        // Read result from device memory
        status = clEnqueueReadBuffer(commandQueue, devBuffC, CL_TRUE, 0, sizeof(float) * c.size(), c.data(), 0,
                                     nullptr, nullptr);
        checkStatus(status, "Could not read the device buffer for the result");

        // Flush the command queue - send commands to device
        status = clFlush(commandQueue);
        checkStatus(status, "Could not flush the command queue");

        // Finish waits for the device to process the commands
        status = clFinish(commandQueue);
        checkStatus(status, "Could not finish on the command queue");

        // Here we can show the processed list
        cout << "One item from result: " << c.at(0) << endl;

        // Release everything
        status = clReleaseKernel(kernel);
        checkStatus(status, "Cannot release kernel");
        status = clReleaseProgram(program);
        checkStatus(status, "Cannot release program");
        status = clReleaseMemObject(devBuffA);
        checkStatus(status, "Cannot allocate memory for vector A");
        status = clReleaseMemObject(devBuffB);
        checkStatus(status, "Cannot allocate memory for vector B");
        status = clReleaseMemObject(devBuffC);
        checkStatus(status, "Cannot allocate memory for vector C");

        status = clReleaseCommandQueue(commandQueue);
        checkStatus(status, "Cannot release command queue");

        status = clReleaseContext(context);
        checkStatus(status, "Cannot release context");
    } catch (const exception& e) {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}