
#pragma once

#include <cublasLt.h>
#include <cublas_size.h>

#ifdef __cplusplus

#include "cuda_offset.h"

static inline void cublasLtMatmulDescSetAttributeParams(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,  const void* buf, size_t sizeInBytes) { /* placeholder */ }

#endif