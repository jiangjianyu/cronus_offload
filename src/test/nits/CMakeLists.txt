cmake_minimum_required(VERSION 3.0)
project(cudaserver LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -g -O0")
# set(CMAKE_CUDA_ARCHITECTURES "75")

file(GLOB srcs
    "test_cuda.cpp"
    "cukernel_only.cu"
)

include_directories(./)
include_directories(/usr/local/cuda-11.4/include)

find_library(CUDART cudart /usr/local/cuda-11.4/targets/x86_64-linux/lib/)
find_library(CUBLAS cublas /usr/local/cuda-11.4/targets/x86_64-linux/lib/)
find_library(CUDADRV cuda /usr/lib/x86_64-linux-gnu/)

add_executable(test_cuda ${srcs})
target_link_libraries(test_cuda ${CUDART})
target_link_libraries(test_cuda dl)
target_link_libraries(test_cuda ${CUDADRV})
set_target_properties(test_cuda PROPERTIES LINK_FLAGS "-Wl,-export-dynamic")

add_executable(test_cublas test_cublas.cpp cukernel_only.cu)
target_link_libraries(test_cublas ${CUDART})
target_link_libraries(test_cublas dl)
target_link_libraries(test_cublas ${CUDADRV})
target_link_libraries(test_cublas ${CUBLAS})
set_target_properties(test_cublas PROPERTIES LINK_FLAGS "-Wl,-export-dynamic")


# add_executable(cukernel_server ${srcs})

# target_link_libraries(cukernel_server ${CUDART})
# target_link_libraries(cukernel_server dl)
# target_link_libraries(cukernel_server cudaserver)

# add_executable(cukernel_client ${srcs_client})
# target_link_libraries(cukernel_client cudart_rpc)

# add_executable(cuclient ${srcs})
