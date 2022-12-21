cmake_minimum_required(VERSION 3.0)
project(cudaserver LANGUAGES C CXX CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
# set(CMAKE_CUDA_ARCHITECTURES "75")

file(GLOB srcs
    "cukernel_only.cu"
)

file(GLOB srcs_client
    "cukernel.cu"
)

include_directories(./)
include_directories(/usr/local/cuda-11.4/include)

find_library(CUDART cudart /usr/local/cuda-11.4/targets/x86_64-linux/lib/)

add_executable(cukernel ${srcs_client})
target_link_libraries(cukernel ${CUDART})
target_link_libraries(cukernel dl)

set_target_properties(cukernel PROPERTIES LINK_FLAGS "-Wl,-export-dynamic")

# add_executable(cukernel_server ${srcs})

# target_link_libraries(cukernel_server ${CUDART})
# target_link_libraries(cukernel_server dl)
# target_link_libraries(cukernel_server cudaserver)

# add_executable(cukernel_client ${srcs_client})
# target_link_libraries(cukernel_client cudart_rpc)

# add_executable(cuclient ${srcs})
