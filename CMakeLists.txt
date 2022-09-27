cmake_minimum_required(VERSION 3.0)
project(cronus_offload)

include_directories(include)

add_subdirectory(src/cudart)
add_subdirectory(src/cudaserver)

