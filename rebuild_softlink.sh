#!/bin/bash

rm -rf src/cudart/cuda_runtime_memory_dispatch.cpp
rm -rf src/cudaserver/cuda_runtime_memory_dispatch.cpp
cd src/cudart
ln -s ../shared/cuda_runtime_memory_dispatch.cpp cuda_runtime_memory_dispatch.cpp
cd ../cudaserver
ln -s ../shared/cuda_runtime_memory_dispatch.cpp cuda_runtime_memory_dispatch.cpp