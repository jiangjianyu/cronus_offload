
#include "cuda_runtime_api.h"
#include "cuda_runtime_u.h"
#include "cuda_runtime_header.h"

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

DEFINE_CUDART_MEMCPY_ALL(cudaMemcpy, (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind), (dst, src, count, kind));
DEFINE_CUDART_MEMCPY_ALL(cudaMemcpy2D, (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind), (dst, dpitch, src, spitch, width, height, kind));
DEFINE_CUDART_MEMCPY_SRC(cudaMemcpy2DToArray, (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind), (dst, wOffset, hOffset, src, spitch, width, height, kind));
DEFINE_CUDART_MEMCPY_DST(cudaMemcpy2DFromArray, (void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind), (dst, dpitch, src, wOffset, hOffset, width, height, kind));
DEFINE_CUDART_MEMCPY_SRC(cudaMemcpyToSymbol, (const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind), ((const char*)symbol, src, count, offset, kind));
DEFINE_CUDART_MEMCPY_DST(cudaMemcpyFromSymbol, (void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind), (dst, (const char*)symbol, count, offset, kind));
DEFINE_CUDART_MEMCPY_ALL(cudaMemcpyAsync, (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream), (dst, src, count, kind, stream));
DEFINE_CUDART_MEMCPY_ALL(cudaMemcpy2DAsync, (void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream), (dst, dpitch, src, spitch, width, height, kind, stream));
DEFINE_CUDART_MEMCPY_SRC(cudaMemcpy2DToArrayAsync, (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream), (dst, wOffset, hOffset, src, spitch, width, height, kind, stream));
DEFINE_CUDART_MEMCPY_DST(cudaMemcpy2DFromArrayAsync, (void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream), (dst, dpitch, src, wOffset, hOffset, width, height, kind, stream));
DEFINE_CUDART_MEMCPY_SRC(cudaMemcpyToSymbolAsync, (const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream), ((const char*) symbol, src, count, offset, kind, stream));
DEFINE_CUDART_MEMCPY_DST(cudaMemcpyFromSymbolAsync, (void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream), (dst, (const char*)symbol, count, offset, kind, stream));

// deprecated
DEFINE_CUDART_MEMCPY_SRC(cudaMemcpyToArray, (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind), (dst, wOffset, hOffset, src, count, kind));
DEFINE_CUDART_MEMCPY_DST(cudaMemcpyFromArray, (void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind), (dst, src, wOffset, hOffset, count, kind));
DEFINE_CUDART_MEMCPY_SRC(cudaMemcpyToArrayAsync, (cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream), (dst, wOffset, hOffset, src, count, kind, stream));
DEFINE_CUDART_MEMCPY_DST(cudaMemcpyFromArrayAsync, (void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream), (dst, src, wOffset, hOffset, count, kind, stream));


// 4 public cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
// 4 public cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
// 2src public cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
// 2dst public cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
// 2src public cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
// 2dst public cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
// 4 public cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
// 4 public cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
// 2src public cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
// 2dst public cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
// 2src public cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
// 2dst public cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);

// deprecated
// 2src public  cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
// 2dst public  cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
// 2src public  cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
// 2dst public  cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);