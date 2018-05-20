#include <iostream>
#include <stdexcept>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 220
#include "CL/cl.h"

using namespace std;

void checkStatus(cl_int status, string message) {
    if (status != CL_SUCCESS) {
        throw runtime_error(message);
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

int main() {

    try {
        vector<float> a, b, c;

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
        cout << "Selected platform id: " << platformId << endl;

        // Get the device list in this platform
        cl_uint num_devices;
        status = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        checkStatus(status, "Cannot query the devices inside the selected platform");

        // Get the device Ids
        vector<cl_device_id> deviceIds (num_devices);
        status = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, num_devices, deviceIds.data(), nullptr);
        printDeviceInfo(deviceIds);

        //

    } catch (const exception& e) {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}