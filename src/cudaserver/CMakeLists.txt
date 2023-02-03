cmake_minimum_required(VERSION 3.9)
project(cudaserver LANGUAGES C CXX CUDA)

file(GLOB srcs
    "*.cpp"
    "hook.c"
    "dispatcher.c"
    "${CMAKE_SOURCE_DIR}/gen/cuda_runtime_t.cpp"
    "${CMAKE_SOURCE_DIR}/gen/cuda_driver_t.cpp"
    "${CMAKE_SOURCE_DIR}/gen/cublas_t.cpp"
    "${CMAKE_SOURCE_DIR}/gen/cublasLt_t.cpp"
)

include_directories(./)
include_directories(/usr/include/libiberty/)
include_directories(/usr/local/cuda-11.4/include)
include_directories(${CMAKE_SOURCE_DIR}/gen/)

find_library(CUDART cudart /usr/local/cuda-11.4/targets/x86_64-linux/lib/)

# add_executable(cudaserver ${srcs})
add_library(cudaserver SHARED ${srcs})
target_link_libraries(cudaserver ${CUDART})
target_link_libraries(cudaserver dl)
target_link_libraries(cudaserver iberty)

# set_target_properties(cudaserver PROPERTIES LINK_FLAGS "-Wl,-export-dynamic")

# use c++14 and cxxabi for abi compatability for pytorch
# see pytorch/torch/share/cmake/Torch/TorchConfig.cmake
set_property(TARGET cudaserver PROPERTY CXX_STANDARD 14)
set_property(TARGET cudaserver PROPERTY INTERFACE_COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=1")