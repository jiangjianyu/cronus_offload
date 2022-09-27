cmake_minimum_required(VERSION 3.9)
project(cudaserver LANGUAGES C CXX CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

file(GLOB srcs
    "*.cpp"
)

include_directories(./)
include_directories(/usr/local/cuda-11.4/include)

find_library(CUDART cudart /usr/local/cuda-11.4/targets/x86_64-linux/lib/)

# add_executable(cudaserver ${srcs})
add_library(cudaserver STATIC ${srcs})
target_link_libraries(cudaserver ${CUDART})
target_link_libraries(cudaserver dl)

set_target_properties(cudaserver PROPERTIES LINK_FLAGS "-Wl,-export-dynamic")
