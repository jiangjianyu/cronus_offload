cmake_minimum_required(VERSION 3.0)
project(cronus_offload)

# add_compile_options(-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=1)

include_directories(include)

execute_process(COMMAND ${CMAKE_SOURCE_DIR}/src/edger8r/linux/out/cronus_edger8r
    ${CMAKE_SOURCE_DIR}/src/edl/cuda_runtime.edl
    --trusted-dir ${CMAKE_SOURCE_DIR}/gen/
    --untrusted-dir ${CMAKE_SOURCE_DIR}/gen/
    --search-path ${CMAKE_SOURCE_DIR}/src/edl/
    )


