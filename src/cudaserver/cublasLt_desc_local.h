
#pragma once

#include <cublasLt.h>
#include <cublas_size.h>

#ifdef __cplusplus

#include "cuda_offset.h"

static inline void cublasLtMatmulDescSetAttributeParams(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,  const void* buf, size_t sizeInBytes) {
    unsigned long r;
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
        case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
            r = *(unsigned long*) buf;
            devOffsetToPtr((void**) buf);
            log_info("transform %lx -> %lx", r, *(unsigned long*) buf);
            break;
        default: break;
    }
}

#endif