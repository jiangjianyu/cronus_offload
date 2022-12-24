cmake_minimum_required(VERSION 3.0)
project(cudaserver LANGUAGES C CXX CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

file(GLOB srcs
    "*.cpp"
    "*.c"
    "${CMAKE_SOURCE_DIR}/gen/cuda_runtime_u.cpp"
)

include_directories(./)
include_directories(/usr/local/cuda-11.4/include)
include_directories(${CMAKE_SOURCE_DIR}/gen/)

find_library(CUDART cudart /usr/local/cuda-11.4/targets/x86_64-linux/lib/)

add_library(cudart_rpc SHARED ${srcs})
# add_executable(cuclient ${srcs})
