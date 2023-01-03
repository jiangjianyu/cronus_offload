#ifndef CUDA_RUNTIME_U_H__
#define CUDA_RUNTIME_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "rpc/rpc.h"
#include "tee_internal_api.h" /* for sgx_satus_t etc. */

#include "cuda_runtime_api.h"

#include <stdlib.h> /* for size_t */

#define TEE_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif


cudaError_t cudaLaunchKernelByName(char* funcname, dim3 gridDim, dim3 blockDim, void* argbuf, int argbufsize, uint32_t* parameters, int partotal_size, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaFuncGetParametersByName(uint32_t* n_par, uint32_t* parameters, const char* entryname, int name_len);
cudaError_t cudaThreadSynchronize();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetValidDevices(int* device_arr, int len);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceFlags(unsigned int* flags);
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority);
cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags);
cudaError_t cudaCtxResetPersistingL2Cache();
cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus);
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId);
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags);
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
cudaError_t cudaMallocArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaFreeArray(cudaArray_t array);
cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost);
cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
cudaError_t cudaMalloc3DArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags);
cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags);
cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p);
cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms* p);
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream);
cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array);
cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx);
cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaArray_t array);
cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap);
cudaError_t cudaMemcpyNone(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpySrc(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpySrcDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
cudaError_t cudaMemcpy2DNone(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DSrc(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DSrcDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DToArrayNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DToArraySrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArrayNone(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArrayDst(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbolNone(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbolSrc(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbolNone(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbolDst(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsyncNone(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyAsyncSrc(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyAsyncDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyAsyncSrcDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsyncNone(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsyncSrc(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsyncDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsyncSrcDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DToArrayAsyncNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DToArrayAsyncSrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DFromArrayAsyncNone(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DFromArrayAsyncDst(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyToSymbolAsyncNone(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyToSymbolAsyncSrc(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbolAsyncNone(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbolAsyncDst(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height);
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol);
cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol);
cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream);
cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void* devPtr, size_t count);
cudaError_t cudaMemcpyToArrayNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyToArraySrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromArrayNone(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromArrayDst(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyToArrayAsyncNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyToArrayAsyncSrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromArrayAsyncNone(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromArrayAsyncDst(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
