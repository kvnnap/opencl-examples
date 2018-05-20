cmake_minimum_required(VERSION 3.10)
project(opencl_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_VERBOSE_MAKEFILE ON)

# OpenCL headers cloned from https://github.com/KhronosGroup/OpenCL-Headers
# C++ bindings available from https://github.com/KhronosGroup/OpenCL-CLHPP/releases.
# Copy cl2.hpp or cl.hpp into the CL directory obtained from the clone of the OpenCL-Headers
# Set OpenCL cache variables for OpenCL to find package
set(OpenCL_INCLUDE_DIR "../OpenCL-Headers")

# Sets OpenCL_INCLUDE_DIRS and OpenCL_LIBRARIES if found
find_package(OpenCL REQUIRED)

# Global include directories
include_directories(${OpenCL_INCLUDE_DIRS})

# Executable
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
