
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#ifdef __cplusplus

static unsigned long __devptr_start = 0x500000000000;

#define NULL_DEV_PTR_OFFSET ((unsigned long)1 << 56)

template<typename T>
static inline void devOffsetToPtr(T* ptr) {
    if (ptr == NULL) return;
    auto val = (unsigned long)*ptr;
    if (val & NULL_DEV_PTR_OFFSET) {
        *ptr = NULL;
    } else if ((unsigned long)val == 0) {
        // if this is a null, then this null is supplied by the app
    } else {
        *ptr = (T)(val + __devptr_start);
    }
}

template<typename T>
static inline void devPtrToOffset(T* offset) {
    if (offset == NULL) return;
    auto val = (unsigned long)*offset;
    if ((unsigned long)val == 0) {
        // do nothing, as this is a zero
    } else {
        *offset = (T)(val - __devptr_start);
    }
}

static inline void handleToStream(cudaStream_t *stream) {

}

static inline void streamToHandle(cudaStream_t *stream) {

}

// cublas
static inline void handleToCublas(cublasHandle_t *handle) {

}

static inline void cublasToHandle(cublasHandle_t *handle) {
    
}

static inline void handleToCublasLt(cublasLtHandle_t *handle) {

}

static inline void cublasLtToHandle(cublasLtHandle_t *handle) {
    
}

// cublasLt

static inline void handleToCublasDesc(cublasLtMatmulDesc_t *handle) {

}

static inline void cublasDescToHandle(cublasLtMatmulDesc_t *handle) {
    
}

static inline void handleToCublasTransformDesc(cublasLtMatrixTransformDesc_t *handle) {

}

static inline void cublasTransformDescToHandle(cublasLtMatrixTransformDesc_t *handle) {
    
}



static inline void handleToCublasLayout(cublasLtMatrixLayout_t *handle) {

}

static inline void cublasLayoutToHandle(cublasLtMatrixLayout_t *handle) {
    
}

static inline void handleToCublasPref(cublasLtMatmulPreference_t *handle) {

}

static inline void cublasPrefToHandle(cublasLtMatmulPreference_t *handle) {
    
}

#endif