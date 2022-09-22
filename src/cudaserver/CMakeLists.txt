cmake_minimum_required(VERSION 3.0)
project(cudaserver)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")

file(GLOB srcs 
    "*.cpp"
)

include_directories(./)
include_directories(include/arch/${ARCH}/)

add_library(cudaserver STATIC ${srcs})