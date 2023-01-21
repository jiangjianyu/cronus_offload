
#pragma once

#include <cublas_v2.h>

static inline int cublasResultSize(const void* par, size_t base_size, cudaDataType resultType) {
	switch (resultType) {
         case CUDA_R_16F:
        case CUDA_C_16F:
        case CUDA_R_16BF:
        case CUDA_C_16BF:   return 2;
        case CUDA_R_32F:
        case CUDA_C_32F:    return 4;
        case CUDA_R_64F:
        case CUDA_C_64F:    return 8;
        case CUDA_R_4I :
        case CUDA_C_4I :
        case CUDA_R_4U :
        case CUDA_C_4U :
        case CUDA_R_8I :
        case CUDA_C_8I :
        case CUDA_R_8U :
        case CUDA_C_8U :    return 1;
        case CUDA_R_16I:
        case CUDA_C_16I:
        case CUDA_R_16U:
        case CUDA_C_16U:    return 2;
        case CUDA_R_32I:
        case CUDA_C_32I:
        case CUDA_R_32U:
        case CUDA_C_32U:    return 4;
        case CUDA_R_64I:
        case CUDA_C_64I:
        case CUDA_R_64U:
        case CUDA_C_64U:    return 8;
        default:
            return 0;
    }
}