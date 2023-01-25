
#pragma once

#include <cublasLt.h>
#include <cublas_size.h>

#ifdef __cplusplus
#include <unordered_map>

typedef struct {
    cublasComputeType_t computeType;
    cudaDataType_t scaleType;
} cublasLtDesc_t;

static std::unordered_map<void*, cublasLtDesc_t> cublasLt_desc_map = std::unordered_map<void*, cublasLtDesc_t>();

static inline void cublasLtMatmulDescInitLogging(cublasLtMatmulDesc_t matmulDesc, size_t size, cublasComputeType_t computeType, cudaDataType_t scaleType) {
    auto desc = cublasLtDesc_t {
        .computeType = computeType,
        .scaleType = scaleType
    };
    cublasLt_desc_map[matmulDesc] = desc;
}

static inline void cublasLtMatmulDescCreateLogging(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType) {
    auto desc = cublasLtDesc_t {
        .computeType = computeType,
        .scaleType = scaleType
    };
    cublasLt_desc_map[*matmulDesc] = desc;
}

void cublasLtMatrixTransformDescInitLogging(cublasLtMatrixTransformDesc_t transformDesc, size_t size, cudaDataType scaleType) {
    auto desc = cublasLtDesc_t {
        .computeType = CUBLAS_COMPUTE_16F,
        .scaleType = scaleType
    };
    cublasLt_desc_map[transformDesc] = desc;
}

static inline int cublasLtMatmulSize(const void* par, size_t base_size, const void* dev_ptr, cublasLtMatmulDesc_t matmulDesc) {
    if (isDevPtr(dev_ptr)) return 0;
    auto desc = cublasLt_desc_map.find(matmulDesc);
    if (desc == cublasLt_desc_map.end()) {
        return 0;
    }
    auto r = cudaTypeSize(desc->second.scaleType);
    return r;
}

static inline int cublasLtMatrixTransformSize(const void* par, size_t base_size, const void* dev_ptr, cublasLtMatrixTransformDesc_t transformDesc) {
    if (isDevPtr(dev_ptr)) return 0;
    auto desc = cublasLt_desc_map.find(transformDesc);
    if (desc == cublasLt_desc_map.end()) {
        return 0;
    }
    auto r = cudaTypeSize(desc->second.scaleType);
    return r;
}

#endif