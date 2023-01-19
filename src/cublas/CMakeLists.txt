cmake_minimum_required(VERSION 3.0)
project(cudaserver LANGUAGES C CXX CUDA)

file(GLOB srcs
    "*.cpp"
    "${CMAKE_SOURCE_DIR}/gen/cublas_u.cpp"
)

include_directories(./)
include_directories(/usr/local/cuda-11.4/include)
include_directories(${CMAKE_SOURCE_DIR}/gen/)

find_library(CUBLAS cublas /usr/local/cuda-11.4/targets/x86_64-linux/lib/)

add_library(cublas SHARED ${srcs})

# use c++14 and cxxabi for abi compatability for pytorch
# see pytorch/torch/share/cmake/Torch/TorchConfig.cmake
set_property(TARGET cublas PROPERTY CXX_STANDARD 14)
set_property(TARGET cublas PROPERTY INTERFACE_COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=1")