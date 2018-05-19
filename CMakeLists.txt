cmake_minimum_required(VERSION 3.10)
project(opencl_hw)

set(CMAKE_CXX_STANDARD 17)


set(OpenCL_INCLUDE_DIR "C:/Program Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.2/include")
set(OpenCL_LIBRARY "C:/Program Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.2/lib/x64/OpenCL.lib")


find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

link_libraries(${OpenCL_LIBRARY})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})