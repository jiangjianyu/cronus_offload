
#pragma once

#define DEFINE_CUDART_MEMCPY_ALL(funcname,desc,call)                                \
    cudaError_t funcname ## None desc { return funcname call; }                     \
    cudaError_t funcname ## Src desc __attribute__((alias(#funcname "None")));      \
    cudaError_t funcname ## Dst desc __attribute__((alias(#funcname "None")));      \
    cudaError_t funcname ## SrcDst desc __attribute__((alias(#funcname "None")));

#define DEFINE_CUDART_MEMCPY_SRC(funcname,desc,call)                                \
    cudaError_t funcname ## None desc { return funcname call; }                     \
    cudaError_t funcname ## Src desc __attribute__((alias(#funcname "None")));

#define DEFINE_CUDART_MEMCPY_DST(funcname,desc,call)                                \
    cudaError_t funcname ## None desc { return funcname call; }                     \
    cudaError_t funcname ## Dst desc __attribute__((alias(#funcname "None")));
