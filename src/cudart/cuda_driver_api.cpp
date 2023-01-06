
#include <cuda.h>

#include <stdlib.h>
#ifdef _MSC_VER
typedef unsigned __int32 cuuint32_t;
typedef unsigned __int64 cuuint64_t;
#else
#include <stdint.h>
typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;
#endif

#if defined(__CUDA_API_VERSION_INTERNAL) || defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
#define __CUDA_DEPRECATED
#elif defined(_MSC_VER)
#define __CUDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __CUDA_DEPRECATED __attribute__((deprecated))
#else
#define __CUDA_DEPRECATED
#endif

#if defined(CUDA_FORCE_API_VERSION)
#error "CUDA_FORCE_API_VERSION is no longer supported."
#endif

#if defined(__CUDA_API_VERSION_INTERNAL) || defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __CUDA_API_PER_THREAD_DEFAULT_STREAM
    #define __CUDA_API_PTDS(api) api ## _ptds
    #define __CUDA_API_PTSZ(api) api ## _ptsz
#else
    #define __CUDA_API_PTDS(api) api
    #define __CUDA_API_PTSZ(api) api
#endif

/**
 * \file cuda.h
 * \brief Header file for the CUDA Toolkit application programming interface.
 *
 * \file cudaGL.h
 * \brief Header file for the OpenGL interoperability functions of the
 * low-level CUDA driver application programming interface.
 *
 * \file cudaD3D9.h
 * \brief Header file for the Direct3D 9 interoperability functions of the
 * low-level CUDA driver application programming interface.
 */

/**
 * \defgroup CUDA_TYPES Data types used by CUDA driver
 * @{
 */

/**
 * CUDA API version number
 */
#define CUDA_VERSION 11040

#ifdef __cplusplus
extern "C" {
#endif

/** @} */ /* END CUDA_TYPES */

#if defined(__GNUC__)
  #if defined(__CUDA_API_PUSH_VISIBILITY_DEFAULT)
    #pragma GCC visibility push(default)
  #endif
#endif

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

/**
 * \defgroup CUDA_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr);
CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr);
/** @} */ /* END CUDA_ERROR */

/**
 * \defgroup CUDA_INITIALIZE Initialization
 *
 * ___MANBRIEF___ initialization functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the initialization functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuInit(unsigned int Flags);
/** @} */ /* END CUDA_INITIALIZE */

/**
 * \defgroup CUDA_VERSION Version Management
 *
 * ___MANBRIEF___ version management functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the version management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuDriverGetVersion(int *driverVersion);
/** @} */ /* END CUDA_VERSION */

/**
 * \defgroup CUDA_DEVICE Device Management
 *
 * ___MANBRIEF___ device management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
CUresult CUDAAPI cuDeviceGetCount(int *count);
CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
CUresult CUDAAPI cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev);
CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult CUDAAPI cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev);
CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags);
CUresult CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev);
CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev);
CUresult CUDAAPI cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope);
/** @} */ /* END CUDA_DEVICE */

/**
 * \defgroup CUDA_DEVICE_DEPRECATED Device Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated device management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
__CUDA_DEPRECATED CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
/** @} */ /* END CUDA_DEVICE_DEPRECATED */

/**
 * \defgroup CUDA_PRIMARY_CTX Primary Context Management
 *
 * ___MANBRIEF___ primary context management functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the primary context management functions of the low-level
 * CUDA driver application programming interface.
 *
 * The primary context is unique per device and shared with the CUDA runtime API.
 * These functions allow integration with other libraries using CUDA.
 *
 * @{
 */
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
/** @} */ /* END CUDA_PRIMARY_CTX */
CUresult CUDAAPI cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev);

/**
 * \defgroup CUDA_CTX Context Management
 *
 * ___MANBRIEF___ context management functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the context management functions of the low-level
 * CUDA driver application programming interface.
 *
 * Please note that some functions are described in
 * \ref CUDA_PRIMARY_CTX "Primary Context Management" section.
 *
 * @{
 */
CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx);
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx);
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx);
CUresult CUDAAPI cuCtxGetDevice(CUdevice *device);
CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags);
CUresult CUDAAPI cuCtxSynchronize(void);
CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value);
CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit);
CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig);
CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config);
CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig);
CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config);
CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
CUresult CUDAAPI cuCtxResetPersistingL2Cache(void);
CUresult CUDAAPI cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type);/** @} */ /* END CUDA_CTX */

/**
 * \defgroup CUDA_CTX_DEPRECATED Context Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated context management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated context management functions of the low-level
 * CUDA driver application programming interface.
 *
 * @{
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags);
__CUDA_DEPRECATED CUresult CUDAAPI cuCtxDetach(CUcontext ctx);/** @} */ /* END CUDA_CTX_DEPRECATED */


/**
 * \defgroup CUDA_MODULE Module Management
 *
 * ___MANBRIEF___ module management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the module management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname);
CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image);
CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
CUresult CUDAAPI cuModuleUnload(CUmodule hmod);
CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name);
CUresult CUDAAPI
cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
CUresult CUDAAPI
cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
    unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult CUDAAPI
cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
    unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult CUDAAPI
cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut);
CUresult CUDAAPI
cuLinkDestroy(CUlinkState state);
/** @} */ /* END CUDA_MODULE */


/**
 * \defgroup CUDA_MEM Memory Management
 *
 * ___MANBRIEF___ memory management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total);
CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
CUresult CUDAAPI cuMemFree(CUdeviceptr dptr);
CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize);
CUresult CUDAAPI cuMemFreeHost(void *p);
CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags);
CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p);
CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId);
CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event);
CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle);
CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr);
CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags);
CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr);
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
CUresult CUDAAPI cuMemHostUnregister(void *p);
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy);
CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy);
CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy);
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy);
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream);
CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);
CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
CUresult CUDAAPI cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
CUresult CUDAAPI cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array);
CUresult CUDAAPI cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap);
CUresult CUDAAPI cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx);
CUresult CUDAAPI cuArrayDestroy(CUarray hArray);
CUresult CUDAAPI cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
CUresult CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels);
CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
/** @} */ /* END CUDA_MEM */

/**
 * \defgroup CUDA_VA Virtual Memory Management
 *
 * ___MANBRIEF___ virtual memory management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the virtual memory management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */

CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult CUDAAPI cuMemMapArrayAsync(CUarrayMapInfo  *mapInfoList, unsigned int count, CUstream hStream);
CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
CUresult CUDAAPI cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr);
CUresult CUDAAPI cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType);
CUresult CUDAAPI cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle);
CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr);
/** @} */ /* END CUDA_VA */

/**
 * \defgroup CUDA_MALLOC_ASYNC Stream Ordered Memory Allocator
 *
 * ___MANBRIEF___ Functions for performing allocation and free operations in stream order.
 *                Functions for controlling the behavior of the underlying allocator.
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream ordered memory allocator exposed by the
 * low-level CUDA driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count);
CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location);
CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps);
CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool);
CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
CUresult CUDAAPI cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags);
CUresult CUDAAPI cuMemPoolImportFromShareableHandle(
        CUmemoryPool *pool_out,
        void *handle,
        CUmemAllocationHandleType handleType,
        unsigned long long flags);
CUresult CUDAAPI cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr);
CUresult CUDAAPI cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData);
/** @} */ /* END CUDA_MALLOC_ASYNC */

/**
 * \defgroup CUDA_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);
CUresult CUDAAPI cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);
/** @} */ /* END CUDA_UNIFIED */

/**
 * \defgroup CUDA_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority);
CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags);
CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);
CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode);
CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph *phGraph);
CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus);
CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
CUresult CUDAAPI cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out,
        cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
CUresult CUDAAPI cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags);
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
CUresult CUDAAPI cuStreamQuery(CUstream hStream);
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
CUresult CUDAAPI cuStreamDestroy(CUstream hStream);
CUresult CUDAAPI cuStreamCopyAttributes(CUstream dst, CUstream src);
CUresult CUDAAPI cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr,
                                      CUstreamAttrValue *value_out);
CUresult CUDAAPI cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                                      const CUstreamAttrValue *value);
/** @} */ /* END CUDA_STREAM */


/**
 * \defgroup CUDA_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);
CUresult CUDAAPI cuEventQuery(CUevent hEvent);
CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
CUresult CUDAAPI cuEventDestroy(CUevent hEvent);
CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
/** @} */ /* END CUDA_EVENT */

/**
 * \defgroup CUDA_EXTRES_INTEROP External Resource Interoperability
 *
 * ___MANBRIEF___ External resource interoperability functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the external resource interoperability functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */

CUresult CUDAAPI cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc);
CUresult CUDAAPI cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc);
CUresult CUDAAPI cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc);
CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory extMem);
CUresult CUDAAPI cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc);
CUresult CUDAAPI cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
CUresult CUDAAPI cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
CUresult CUDAAPI cuDestroyExternalSemaphore(CUexternalSemaphore extSem);
/** @} */ /* END CUDA_EXTRES_INTEROP */

/**
 * \defgroup CUDA_MEMOP Stream memory operations
 *
 * ___MANBRIEF___ Stream memory operations of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream memory operations of the low-level CUDA
 * driver application programming interface.
 *
 * The whole set of operations is disabled by default. Users are required
 * to explicitly enable them, e.g. on Linux by passing the kernel module
 * parameter shown below:
 *     modprobe nvidia NVreg_EnableStreamMemOPs=1
 * There is currently no way to enable these operations on other operating
 * systems.
 *
 * Users can programmatically query whether the device supports these
 * operations with ::cuDeviceGetAttribute() and
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS.
 *
 * Support for the ::CU_STREAM_WAIT_VALUE_NOR flag can be queried with
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.
 *
 * Support for the ::cuStreamWriteValue64() and ::cuStreamWaitValue64()
 * functions, as well as for the ::CU_STREAM_MEM_OP_WAIT_VALUE_64 and
 * ::CU_STREAM_MEM_OP_WRITE_VALUE_64 flags, can be queried with
 * ::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.
 *
 * Support for both ::CU_STREAM_WAIT_VALUE_FLUSH and
 * ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated platform
 * hardware features and can be queried with ::cuDeviceGetAttribute() and
 * ::CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.
 *
 * Note that all memory pointers passed as parameters to these operations
 * are device pointers. Where necessary a device pointer should be
 * obtained, for example with ::cuMemHostGetDevicePointer().
 *
 * None of the operations accepts pointers to managed memory buffers
 * (::cuMemAllocManaged).
 *
 * @{
 */
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);
/** @} */ /* END CUDA_MEMOP */

/**
 * \defgroup CUDA_EXEC Execution Control
 *
 * ___MANBRIEF___ execution control functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the execution control functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);
CUresult CUDAAPI cuFuncGetModule(CUmodule *hmod, CUfunction hfunc);
CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra);
CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams);
__CUDA_DEPRECATED CUresult CUDAAPI cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags);
CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData);
/** @} */ /* END CUDA_EXEC */

/**
 * \defgroup CUDA_EXEC_DEPRECATED Execution Control [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated execution control functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated execution control functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
__CUDA_DEPRECATED CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
__CUDA_DEPRECATED CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset, unsigned int value);
__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset, float value);
__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
__CUDA_DEPRECATED CUresult CUDAAPI cuLaunch(CUfunction f);
__CUDA_DEPRECATED CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
__CUDA_DEPRECATED CUresult CUDAAPI cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
__CUDA_DEPRECATED CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);/** @} */ /* END CUDA_EXEC_DEPRECATED */

/**
 * \defgroup CUDA_GRAPH Graph Management
 *
 * ___MANBRIEF___ graph management functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graph management functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuGraphCreate(CUgraph *phGraph, unsigned int flags);
CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
CUresult CUDAAPI cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams);
CUresult CUDAAPI cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams);
CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
CUresult CUDAAPI cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph);
CUresult CUDAAPI cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph);
CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies);
CUresult CUDAAPI cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
 CUresult CUDAAPI cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
CUresult CUDAAPI cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event);
CUresult CUDAAPI cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
CUresult CUDAAPI cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
CUresult CUDAAPI cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event);
CUresult CUDAAPI cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out);
CUresult CUDAAPI cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out);
CUresult CUDAAPI cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out);
CUresult CUDAAPI cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr);
CUresult CUDAAPI cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out);
CUresult CUDAAPI cuDeviceGraphMemTrim(CUdevice device);
CUresult CUDAAPI cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value);
CUresult CUDAAPI cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value);
CUresult CUDAAPI cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph);
CUresult CUDAAPI cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type);
CUresult CUDAAPI cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
CUresult CUDAAPI cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes);
CUresult CUDAAPI cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges);
CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies);
CUresult CUDAAPI cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes);
CUresult CUDAAPI cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
CUresult CUDAAPI cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
CUresult CUDAAPI cuGraphDestroyNode(CUgraphNode hNode);
CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags);
CUresult CUDAAPI cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
CUresult CUDAAPI cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
CUresult CUDAAPI cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph);
CUresult CUDAAPI cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
CUresult CUDAAPI cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
CUresult CUDAAPI cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);
CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec);
CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph);
CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out);
CUresult CUDAAPI cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src);
CUresult CUDAAPI cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                      CUkernelNodeAttrValue *value_out);
CUresult CUDAAPI cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                      const CUkernelNodeAttrValue *value);
CUresult CUDAAPI cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags);
CUresult CUDAAPI cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy,
                                    unsigned int initialRefcount, unsigned int flags);
CUresult CUDAAPI cuUserObjectRetain(CUuserObject object, unsigned int count);
CUresult CUDAAPI cuUserObjectRelease(CUuserObject object, unsigned int count);
CUresult CUDAAPI cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
CUresult CUDAAPI cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count);
/** @} */ /* END CUDA_GRAPH */

/**
 * \defgroup CUDA_OCCUPANCY Occupancy
 *
 * ___MANBRIEF___ occupancy calculation functions of the low-level CUDA driver
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the occupancy calculation functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
CUresult CUDAAPI cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
/** @} */ /* END CUDA_OCCUPANCY */

/**
 * \defgroup CUDA_TEXREF_DEPRECATED Texture Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated texture reference management functions of the
 * low-level CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the deprecated texture reference management
 * functions of the low-level CUDA driver application programming interface.
 *
 * @{
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefCreate(CUtexref *pTexRef);
__CUDA_DEPRECATED CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef);
/** @} */ /* END CUDA_TEXREF_DEPRECATED */


/**
 * \defgroup CUDA_SURFREF_DEPRECATED Surface Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ surface reference management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the surface reference management functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
__CUDA_DEPRECATED CUresult CUDAAPI cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef);
/** @} */ /* END CUDA_SURFREF_DEPRECATED */

/**
 * \defgroup CUDA_TEXOBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the texture object management functions of the
 * low-level CUDA driver application programming interface. The texture
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */
CUresult CUDAAPI cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc);
CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject);
CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject);
CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject);
CUresult CUDAAPI cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject);
/** @} */ /* END CUDA_TEXOBJECT */

/**
 * \defgroup CUDA_SURFOBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the surface object management functions of the
 * low-level CUDA driver application programming interface. The surface
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */
CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc);
CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject);
CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject);
/** @} */ /* END CUDA_SURFOBJECT */

/**
 * \defgroup CUDA_PEER_ACCESS Peer Context Memory Access
 *
 * ___MANBRIEF___ direct peer context memory access functions of the low-level
 * CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the direct peer context memory access functions
 * of the low-level CUDA driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext);
CUresult CUDAAPI cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);
/** @} */ /* END CUDA_PEER_ACCESS */

/**
 * \defgroup CUDA_GRAPHICS Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the
 * low-level CUDA driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource);
CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource);
CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource);
CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags);
CUresult CUDAAPI cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
/** @} */ /* END CUDA_GRAPHICS */

/**
 * \defgroup CUDA_DRIVER_ENTRY_POINT Driver Entry Point Access 
 *
 * ___MANBRIEF___ driver entry point access functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the driver entry point access functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);
/** @} */ /* END CUDA_DRIVER_ENTRY_POINT */

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId);

/**
 * CUDA API versioning support
 */
#if defined(__CUDA_API_VERSION_INTERNAL)
    #undef cuMemHostRegister
    #undef cuGraphicsResourceSetMapFlags
    #undef cuLinkCreate
    #undef cuLinkAddData
    #undef cuLinkAddFile
    #undef cuDeviceTotalMem
    #undef cuCtxCreate
    #undef cuModuleGetGlobal
    #undef cuMemGetInfo
    #undef cuMemAlloc
    #undef cuMemAllocPitch
    #undef cuMemFree
    #undef cuMemGetAddressRange
    #undef cuMemAllocHost
    #undef cuMemHostGetDevicePointer
    #undef cuMemcpyHtoD
    #undef cuMemcpyDtoH
    #undef cuMemcpyDtoD
    #undef cuMemcpyDtoA
    #undef cuMemcpyAtoD
    #undef cuMemcpyHtoA
    #undef cuMemcpyAtoH
    #undef cuMemcpyAtoA
    #undef cuMemcpyHtoAAsync
    #undef cuMemcpyAtoHAsync
    #undef cuMemcpy2D
    #undef cuMemcpy2DUnaligned
    #undef cuMemcpy3D
    #undef cuMemcpyHtoDAsync
    #undef cuMemcpyDtoHAsync
    #undef cuMemcpyDtoDAsync
    #undef cuMemcpy2DAsync
    #undef cuMemcpy3DAsync
    #undef cuMemsetD8
    #undef cuMemsetD16
    #undef cuMemsetD32
    #undef cuMemsetD2D8
    #undef cuMemsetD2D16
    #undef cuMemsetD2D32
    #undef cuArrayCreate
    #undef cuArrayGetDescriptor
    #undef cuArray3DCreate
    #undef cuArray3DGetDescriptor
    #undef cuTexRefSetAddress
    #undef cuTexRefSetAddress2D
    #undef cuTexRefGetAddress
    #undef cuGraphicsResourceGetMappedPointer
    #undef cuCtxDestroy
    #undef cuCtxPopCurrent
    #undef cuCtxPushCurrent
    #undef cuStreamDestroy
    #undef cuEventDestroy
    #undef cuMemcpy
    #undef cuMemcpyAsync
    #undef cuMemcpyPeer
    #undef cuMemcpyPeerAsync
    #undef cuMemcpy3DPeer
    #undef cuMemcpy3DPeerAsync
    #undef cuMemsetD8Async
    #undef cuMemsetD16Async
    #undef cuMemsetD32Async
    #undef cuMemsetD2D8Async
    #undef cuMemsetD2D16Async
    #undef cuMemsetD2D32Async
    #undef cuStreamGetPriority
    #undef cuStreamGetFlags
    #undef cuStreamGetCtx
    #undef cuStreamWaitEvent
    #undef cuStreamAddCallback
    #undef cuStreamAttachMemAsync
    #undef cuStreamQuery
    #undef cuStreamSynchronize
    #undef cuEventRecord
    #undef cuEventRecordWithFlags
    #undef cuLaunchKernel
    #undef cuLaunchHostFunc
    #undef cuGraphicsMapResources
    #undef cuGraphicsUnmapResources
    #undef cuStreamWriteValue32
    #undef cuStreamWaitValue32
    #undef cuStreamWriteValue64
    #undef cuStreamWaitValue64
    #undef cuStreamBatchMemOp
    #undef cuMemPrefetchAsync
    #undef cuLaunchCooperativeKernel
    #undef cuSignalExternalSemaphoresAsync
    #undef cuWaitExternalSemaphoresAsync
    #undef cuStreamBeginCapture
    #undef cuStreamEndCapture
    #undef cuStreamIsCapturing
    #undef cuStreamGetCaptureInfo
    #undef cuStreamGetCaptureInfo_v2
    #undef cuGraphUpload
    #undef cuGraphLaunch
    #undef cuDevicePrimaryCtxRelease
    #undef cuDevicePrimaryCtxReset
    #undef cuDevicePrimaryCtxSetFlags
    #undef cuIpcOpenMemHandle
    #undef cuStreamCopyAttributes
    #undef cuStreamSetAttribute
    #undef cuStreamGetAttribute
    #undef cuGraphInstantiate
    #undef cuMemMapArrayAsync
    #undef cuMemFreeAsync 
    #undef cuMemAllocAsync 
    #undef cuMemAllocFromPoolAsync 
    #undef cuStreamUpdateCaptureDependencies

    CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
    CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags);
    CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
    CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
        unsigned int numOptions, CUjit_option *options, void **optionValues);
    CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
        unsigned int numOptions, CUjit_option *options, void **optionValues);
    CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);

    typedef unsigned int CUdeviceptr_v1;

    typedef struct CUDA_MEMCPY2D_v1_st
    {
        unsigned int srcXInBytes;   /**< Source X in bytes */
        unsigned int srcY;          /**< Source Y */
        CUmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
        const void *srcHost;        /**< Source host pointer */
        CUdeviceptr_v1 srcDevice;   /**< Source device pointer */
        CUarray srcArray;           /**< Source array reference */
        unsigned int srcPitch;      /**< Source pitch (ignored when src is array) */

        unsigned int dstXInBytes;   /**< Destination X in bytes */
        unsigned int dstY;          /**< Destination Y */
        CUmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
        void *dstHost;              /**< Destination host pointer */
        CUdeviceptr_v1 dstDevice;   /**< Destination device pointer */
        CUarray dstArray;           /**< Destination array reference */
        unsigned int dstPitch;      /**< Destination pitch (ignored when dst is array) */

        unsigned int WidthInBytes;  /**< Width of 2D memory copy in bytes */
        unsigned int Height;        /**< Height of 2D memory copy */
    } CUDA_MEMCPY2D_v1;

    typedef struct CUDA_MEMCPY3D_v1_st
    {
        unsigned int srcXInBytes;   /**< Source X in bytes */
        unsigned int srcY;          /**< Source Y */
        unsigned int srcZ;          /**< Source Z */
        unsigned int srcLOD;        /**< Source LOD */
        CUmemorytype srcMemoryType; /**< Source memory type (host, device, array) */
        const void *srcHost;        /**< Source host pointer */
        CUdeviceptr_v1 srcDevice;   /**< Source device pointer */
        CUarray srcArray;           /**< Source array reference */
        void *reserved0;            /**< Must be NULL */
        unsigned int srcPitch;      /**< Source pitch (ignored when src is array) */
        unsigned int srcHeight;     /**< Source height (ignored when src is array; may be 0 if Depth==1) */

        unsigned int dstXInBytes;   /**< Destination X in bytes */
        unsigned int dstY;          /**< Destination Y */
        unsigned int dstZ;          /**< Destination Z */
        unsigned int dstLOD;        /**< Destination LOD */
        CUmemorytype dstMemoryType; /**< Destination memory type (host, device, array) */
        void *dstHost;              /**< Destination host pointer */
        CUdeviceptr_v1 dstDevice;   /**< Destination device pointer */
        CUarray dstArray;           /**< Destination array reference */
        void *reserved1;            /**< Must be NULL */
        unsigned int dstPitch;      /**< Destination pitch (ignored when dst is array) */
        unsigned int dstHeight;     /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

        unsigned int WidthInBytes;  /**< Width of 3D memory copy in bytes */
        unsigned int Height;        /**< Height of 3D memory copy */
        unsigned int Depth;         /**< Depth of 3D memory copy */
    } CUDA_MEMCPY3D_v1;

    typedef struct CUDA_ARRAY_DESCRIPTOR_v1_st
    {
        unsigned int Width;         /**< Width of array */
        unsigned int Height;        /**< Height of array */

        CUarray_format Format;      /**< Array format */
        unsigned int NumChannels;   /**< Channels per array element */
    } CUDA_ARRAY_DESCRIPTOR_v1;

    typedef struct CUDA_ARRAY3D_DESCRIPTOR_v1_st
    {
        unsigned int Width;         /**< Width of 3D array */
        unsigned int Height;        /**< Height of 3D array */
        unsigned int Depth;         /**< Depth of 3D array */

        CUarray_format Format;      /**< Array format */
        unsigned int NumChannels;   /**< Channels per array element */
        unsigned int Flags;         /**< Flags */
    } CUDA_ARRAY3D_DESCRIPTOR_v1;

    CUresult CUDAAPI cuDeviceTotalMem(unsigned int *bytes, CUdevice dev);
    CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
    CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr_v1 *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
    CUresult CUDAAPI cuMemGetInfo(unsigned int *free, unsigned int *total);
    CUresult CUDAAPI cuMemAlloc(CUdeviceptr_v1 *dptr, unsigned int bytesize);
    CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr_v1 *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
    CUresult CUDAAPI cuMemFree(CUdeviceptr_v1 dptr);
    CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr_v1 *pbase, unsigned int *psize, CUdeviceptr_v1 dptr);
    CUresult CUDAAPI cuMemAllocHost(void **pp, unsigned int bytesize);
    CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr_v1 *pdptr, void *p, unsigned int Flags);
    CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, unsigned int dstOffset, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr_v1 dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D_v1 *pCopy);
    CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D_v1 *pCopy);
    CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D_v1 *pCopy);
    CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D_v1 *pCopy, CUstream hStream);
    CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D_v1 *pCopy, CUstream hStream);
    CUresult CUDAAPI cuMemsetD8(CUdeviceptr_v1 dstDevice, unsigned char uc, unsigned int N);
    CUresult CUDAAPI cuMemsetD16(CUdeviceptr_v1 dstDevice, unsigned short us, unsigned int N);
    CUresult CUDAAPI cuMemsetD32(CUdeviceptr_v1 dstDevice, unsigned int ui, unsigned int N);
    CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height);
    CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height);
    CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height);
    CUresult CUDAAPI cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray);
    CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR_v1 *pArrayDescriptor, CUarray hArray);
    CUresult CUDAAPI cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR_v1 *pAllocateArray);
    CUresult CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR_v1 *pArrayDescriptor, CUarray hArray);
    CUresult CUDAAPI cuTexRefSetAddress(unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr_v1 dptr, unsigned int bytes);
    CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR_v1 *desc, CUdeviceptr_v1 dptr, unsigned int Pitch);
    CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr_v1 *pdptr, CUtexref hTexRef);
    CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr_v1 *pDevPtr, unsigned int *pSize, CUgraphicsResource resource);

    CUresult CUDAAPI cuCtxDestroy(CUcontext ctx);
    CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx);
    CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
    CUresult CUDAAPI cuStreamDestroy(CUstream hStream);
    CUresult CUDAAPI cuEventDestroy(CUevent hEvent);
    CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
    CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
    CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);

    CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy);
    CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy);
    CUresult CUDAAPI cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy);
    CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
    CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
    CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
    CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N);
    CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
    CUresult CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
    CUresult CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
    CUresult CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
    CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
    CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
    CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy);
    CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream);

    CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
    CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
    CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
    CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
    CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
    CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);

    CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority);
    CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags);
    CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
    CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
    CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
    CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
    CUresult CUDAAPI cuStreamQuery(CUstream hStream);
    CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
    CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
    CUresult CUDAAPI cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);
    CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
    CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData);
    CUresult CUDAAPI cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
    CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
    CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
    CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
    CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
    CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
    CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);
    CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
    CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
    CUresult CUDAAPI cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
    CUresult CUDAAPI cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
    CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
    CUresult CUDAAPI cuStreamBeginCapture_ptsz(CUstream hStream);
    CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);
    CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph *phGraph);
    CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus);
    CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
    CUresult CUDAAPI cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
    CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraph, CUstream hStream);
    CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraph, CUstream hStream);
    CUresult CUDAAPI cuStreamCopyAttributes(CUstream dstStream, CUstream srcStream);
    CUresult CUDAAPI cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value);
    CUresult CUDAAPI cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *param);

    CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags);
    CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
    CUresult CUDAAPI cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream);

    CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
    CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
    CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);

    CUresult CUDAAPI cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags);
#elif defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)
static inline CUresult cuGetProcAddress_ptsz(const char *symbol, void **funcPtr, int driverVersion, cuuint64_t flags) {
    const int procAddressMask = (CU_GET_PROC_ADDRESS_LEGACY_STREAM|
                                 CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM);
    if ((flags & procAddressMask) == 0) {
        flags |= CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
    }
    return cuGetProcAddress(symbol, funcPtr, driverVersion, flags); 
}
#define cuGetProcAddress cuGetProcAddress_ptsz
#endif

#ifdef __cplusplus
}
#endif

#if defined(__GNUC__)
  #if defined(__CUDA_API_PUSH_VISIBILITY_DEFAULT)
    #pragma GCC visibility pop
  #endif
#endif

#undef __CUDA_DEPRECATED