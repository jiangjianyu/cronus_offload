
#include "cuda_runtime_api.h"
#include "cuda_runtime_header.h"

#define __CUDA_DEPRECATED
#define __dv(v)

#define CUDARTAPI_OLD CUDARTAPI
#undef CUDARTAPI
#define CUDARTAPI __attribute__((weak))

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \defgroup CUDART_DEVICE Device Management
 *
 * ___MANBRIEF___ device management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceReset(void) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11030
extern __host__ cudaError_t CUDARTAPI cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope) CUDART_NOT_IMPLEMENTED;
#endif

/**
 * \defgroup CUDART_THREAD_DEPRECATED Thread Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated thread management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated thread management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadExit(void) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSetLimit(enum cudaLimit limit, size_t value) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error) { cudart_not_implemented(NULL); }
extern __host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error) { cudart_not_implemented(NULL); }
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device, int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags ) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetDeviceFlags( unsigned int *flags ) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream, int *priority) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache(void) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) CUDART_NOT_IMPLEMENTED;
// extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetAttribute(
//         cudaStream_t hStream, enum cudaStreamAttrID attr,
//         union cudaStreamAttrValue *value_out) CUDART_NOT_IMPLEMENTED;
// extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamSetAttribute(
//         cudaStream_t hStream, enum cudaStreamAttrID attr,
//         const union cudaStreamAttrValue *value) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags __dv(0)) CUDART_NOT_IMPLEMENTED;
typedef void (CUDART_CB *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);
extern __host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) CUDART_NOT_IMPLEMENTED;

// #if defined(__cplusplus)
// extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags = cudaMemAttachSingle) CUDART_NOT_IMPLEMENTED;
// #else
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags) CUDART_NOT_IMPLEMENTED;
// #endif

extern __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), cudaGraph_t *graph_out __dv(0), const cudaGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags __dv(0)) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream __dv(0), unsigned int flags __dv(0)) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_EXTRES_INTEROP External Resource Interoperability
 *
 * ___MANBRIEF___ External resource interoperability functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the external resource interoperability functions of the CUDA
 * runtime application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t extMem) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) CUDART_NOT_IMPLEMENTED;


/**
 * \defgroup CUDART_EXECUTION Execution Control
 *
 * ___MANBRIEF___ execution control functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the execution control functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED  __host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_OCCUPANCY Occupancy
 *
 * ___MANBRIEF___ occupancy calculation functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the occupancy calculation functions of the CUDA runtime
 * application programming interface.
 *
 * Besides the occupancy calculator functions
 * (\ref ::cudaOccupancyMaxActiveBlocksPerMultiprocessor and \ref ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags),
 * there are also C++ only occupancy-based launch configuration functions documented in
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * See
 * \ref ::cudaOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)"
 * \ref ::cudaOccupancyAvailableDynamicSMemPerBlock(size_t*, T, int, int) "cudaOccupancyAvailableDynamicSMemPerBlock (C++ API)",
 *
 * @{
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_MEMORY Memory Management
 *
 * ___MANBRIEF___ memory management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */
// #if defined(__cplusplus)
// extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal) CUDART_NOT_IMPLEMENTED;
// #else
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) CUDART_NOT_IMPLEMENTED;
// #endif

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_MEMORY_DEPRECATED Memory Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated memory management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated memory management functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_MEMORY_POOLS Stream Ordered Memory Allocator 
 *
 * ___MANBRIEF___ Functions for performing allocation and free operations in stream order.
 *                Functions for controlling the behavior of the underlying allocator.
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 * 
 *
 * @{
 *
 * \section CUDART_MEMORY_POOLS_overview overview
 *
 * The asynchronous allocator allows the user to allocate and free in stream order.
 * All asynchronous accesses of the allocation must happen between
 * the stream executions of the allocation and the free. If the memory is accessed
 * outside of the promised stream order, a use before allocation / use after free error
 * will cause undefined behavior.
 *
 * The allocator is free to reallocate the memory as long as it can guarantee
 * that compliant memory accesses will not overlap temporally.
 * The allocator may refer to internal stream ordering as well as inter-stream dependencies
 * (such as CUDA events and null stream dependencies) when establishing the temporal guarantee.
 * The allocator may also insert inter-stream dependencies to establish the temporal guarantee.
 *
 * \section CUDART_MEMORY_POOLS_support Supported Platforms
 *
 * Whether or not a device supports the integrated stream ordered memory allocator
 * may be queried by calling ::cudaDeviceGetAttribute() with the device attribute
 * ::cudaDevAttrMemoryPoolsSupported.
 */
extern __host__ cudaError_t CUDARTAPI cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaFreeAsync(void *devPtr, cudaStream_t hStream) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value ) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool, struct cudaMemLocation *location) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *poolProps) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t memPool) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
//     void                            *shareableHandle,
//     cudaMemPool_t                    memPool,
//     enum cudaMemAllocationHandleType handleType,
//     unsigned int                     flags) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
//     cudaMemPool_t                   *memPool,
//     void                            *shareableHandle,
//     enum cudaMemAllocationHandleType handleType,
//     unsigned int                     flags) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData, void *ptr) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData *exportData) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the CUDA 
 * runtime application programming interface.
 *
 * @{
 *
 * \section CUDART_UNIFIED_overview Overview
 *
 * CUDA devices can share a unified address space with the host.  
 * For these devices there is no distinction between a device
 * pointer and a host pointer -- the same pointer value may be 
 * used to access memory from the host program and from a kernel 
 * running on the device (with exceptions enumerated below).
 *
 * \section CUDART_UNIFIED_support Supported Platforms
 * 
 * Whether or not a device supports unified addressing may be 
 * queried by calling ::cudaGetDeviceProperties() with the device 
 * property ::cudaDeviceProp::unifiedAddressing.
 *
 * Unified addressing is automatically enabled in 64-bit processes .
 *
 * \section CUDART_UNIFIED_lookup Looking Up Information from Pointer Values
 *
 * It is possible to look up information about the memory which backs a 
 * pointer value.  For instance, one may want to know if a pointer points
 * to host or device memory.  As another example, in the case of device 
 * memory, one may want to know on which CUDA device the memory 
 * resides.  These properties may be queried using the function 
 * ::cudaPointerGetAttributes()
 *
 * Since pointers are unique, it is not necessary to specify information
 * about the pointers specified to ::cudaMemcpy() and other copy functions.  
 * The copy direction ::cudaMemcpyDefault may be used to specify that the 
 * CUDA runtime should infer the location of the pointer from its value.
 *
 * \section CUDART_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
 *
 * All host memory allocated through all devices using ::cudaMallocHost() and
 * ::cudaHostAlloc() is always directly accessible from all devices that 
 * support unified addressing.  This is the case regardless of whether or 
 * not the flags ::cudaHostAllocPortable and ::cudaHostAllocMapped are 
 * specified.
 *
 * The pointer value through which allocated host memory may be accessed 
 * in kernels on all devices that support unified addressing is the same 
 * as the pointer value through which that memory is accessed on the host.
 * It is not necessary to call ::cudaHostGetDevicePointer() to get the device 
 * pointer for these allocations.  
 *
 * Note that this is not the case for memory allocated using the flag
 * ::cudaHostAllocWriteCombined, as discussed below.
 *
 * \section CUDART_UNIFIED_autopeerregister Direct Access of Peer Memory
 
 * Upon enabling direct access from a device that supports unified addressing 
 * to another peer device that supports unified addressing using 
 * ::cudaDeviceEnablePeerAccess() all memory allocated in the peer device using 
 * ::cudaMalloc() and ::cudaMallocPitch() will immediately be accessible 
 * by the current device.  The device pointer value through 
 * which any peer's memory may be accessed in the current device 
 * is the same pointer value through which that memory may be 
 * accessed from the peer device. 
 *
 * \section CUDART_UNIFIED_exceptions Exceptions, Disjoint Addressing
 * 
 * Not all memory may be accessed on devices through the same pointer
 * value through which they are accessed on the host.  These exceptions
 * are host memory registered using ::cudaHostRegister() and host memory
 * allocated using the flag ::cudaHostAllocWriteCombined.  For these 
 * exceptions, there exists a distinct host and device address for the
 * memory.  The device address is guaranteed to not overlap any valid host
 * pointer range and is guaranteed to have the same value across all devices
 * that support unified addressing.  
 * 
 * This device address may be queried using ::cudaHostGetDevicePointer() 
 * when a device using unified addressing is current.  Either the host 
 * or the unified device pointer value may be used to refer to this memory 
 * in ::cudaMemcpy() and similar functions using the ::cudaMemcpyDefault 
 * memory direction.
 *
 */
extern __host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_PEER Peer Device Memory Access
 *
 * ___MANBRIEF___ peer device memory access functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the peer device memory access functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice) CUDART_NOT_IMPLEMENTED;

/** \defgroup CUDART_OPENGL OpenGL Interoperability */

/** \defgroup CUDART_OPENGL_DEPRECATED OpenGL Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D9 Direct3D 9 Interoperability */

/** \defgroup CUDART_D3D9_DEPRECATED Direct3D 9 Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D10 Direct3D 10 Interoperability */

/** \defgroup CUDART_D3D10_DEPRECATED Direct3D 10 Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D11 Direct3D 11 Interoperability */

/** \defgroup CUDART_D3D11_DEPRECATED Direct3D 11 Interoperability [DEPRECATED] */

/** \defgroup CUDART_VDPAU VDPAU Interoperability */

/** \defgroup CUDART_EGL EGL Interoperability */

/**
 * \defgroup CUDART_INTEROP Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the CUDA
 * runtime application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0)) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_TEXTURE Texture Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ texture reference management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture reference management functions
 * of the CUDA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX)) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const void *symbol) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_SURFACE Surface Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ surface reference management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level surface reference management functions
 * of the CUDA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) CUDART_NOT_IMPLEMENTED;
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_TEXTURE_OBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the CUDA runtime application programming interface. The texture
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array) CUDART_NOT_IMPLEMENTED;
extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) { cudart_not_implemented_noreturn; }
extern __host__ cudaError_t CUDARTAPI cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDestroyTextureObject(cudaTextureObject_t texObject) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_SURFACE_OBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the CUDA runtime application programming interface. The surface object 
 * API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART__VERSION Version Management
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) CUDART_NOT_IMPLEMENTED;
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_GRAPH Graph Management
 *
 * ___MANBRIEF___ graph management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graph management functions of CUDA
 * runtime application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeCopyAttributes(
        cudaGraphNode_t hSrc,
        cudaGraphNode_t hDst) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetAttribute(
//     cudaGraphNode_t hNode,
//     enum cudaKernelNodeAttrID attr,
//     union cudaKernelNodeAttrValue *value_out) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetAttribute(
//     cudaGraphNode_t hNode,
//     enum cudaKernelNodeAttrID attr,
//     const union cudaKernelNodeAttrValue *value) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNodeToSymbol(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNodeFromSymbol(
    cudaGraphNode_t* pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies,
    size_t numDependencies,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode1D(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParamsToSymbol(
    cudaGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParamsFromSymbol(
    cudaGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams1D(
    cudaGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams *nodeParams) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams *params_out) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaDeviceGraphMemTrim(int device) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphDestroyNode(cudaGraphNode_t node) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParamsToSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParams1D(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) CUDART_NOT_IMPLEMENTED;
#endif

#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, enum cudaGraphExecUpdateResult *updateResult_out) CUDART_NOT_IMPLEMENTED;

#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) CUDART_NOT_IMPLEMENTED;
#endif

extern __host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaUserObjectRetain(cudaUserObject_t object, unsigned int count __dv(1)) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaUserObjectRelease(cudaUserObject_t object, unsigned int count __dv(1)) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count __dv(1), unsigned int flags __dv(0)) CUDART_NOT_IMPLEMENTED;
// extern __host__ cudaError_t CUDARTAPI cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count __dv(1)) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_DRIVER_ENTRY_POINT Driver Entry Point Access
 *
 * ___MANBRIEF___ driver entry point access functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the driver entry point access functions of CUDA
 * runtime application programming interface.
 *
 * @{
 */
extern __host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags) CUDART_NOT_IMPLEMENTED;
extern __host__ cudaError_t CUDARTAPI cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId) CUDART_NOT_IMPLEMENTED;

/**
 * \defgroup CUDART_HIGHLEVEL C++ API Routines
 *
 * ___MANBRIEF___ C++ high level API functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the C++ high level API functions of the CUDA runtime
 * application programming interface. To use these functions, your
 * application needs to be compiled with the \p nvcc compiler.
 *
 * \brief C++-style interface built on top of CUDA runtime API
 */

/**
 * \defgroup CUDART_DRIVER Interactions with the CUDA Driver API
 *
 * ___MANBRIEF___ interactions between CUDA Driver API and CUDA Runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the interactions between the CUDA Driver API and the CUDA Runtime API
 *
 * @{
 *
 * \section CUDART_CUDA_primary Primary Contexts
 *
 * There exists a one to one relationship between CUDA devices in the CUDA Runtime
 * API and ::CUcontext s in the CUDA Driver API within a process.  The specific
 * context which the CUDA Runtime API uses for a device is called the device's
 * primary context.  From the perspective of the CUDA Runtime API, a device and 
 * its primary context are synonymous.
 *
 * \section CUDART_CUDA_init Initialization and Tear-Down
 *
 * CUDA Runtime API calls operate on the CUDA Driver API ::CUcontext which is current to
 * to the calling host thread.  
 *
 * The function ::cudaSetDevice() makes the primary context for the
 * specified device current to the calling thread by calling ::cuCtxSetCurrent().
 *
 * The CUDA Runtime API will automatically initialize the primary context for
 * a device at the first CUDA Runtime API call which requires an active context.
 * If no ::CUcontext is current to the calling thread when a CUDA Runtime API call 
 * which requires an active context is made, then the primary context for a device 
 * will be selected, made current to the calling thread, and initialized.
 *
 * The context which the CUDA Runtime API initializes will be initialized using 
 * the parameters specified by the CUDA Runtime API functions
 * ::cudaSetDeviceFlags(), 
 * ::cudaD3D9SetDirect3DDevice(), 
 * ::cudaD3D10SetDirect3DDevice(), 
 * ::cudaD3D11SetDirect3DDevice(), 
 * ::cudaGLSetGLDevice(), and
 * ::cudaVDPAUSetVDPAUDevice().
 * Note that these functions will fail with ::cudaErrorSetOnActiveProcess if they are 
 * called when the primary context for the specified device has already been initialized.
 * (or if the current device has already been initialized, in the case of 
 * ::cudaSetDeviceFlags()). 
 *
 * Primary contexts will remain active until they are explicitly deinitialized 
 * using ::cudaDeviceReset().  The function ::cudaDeviceReset() will deinitialize the 
 * primary context for the calling thread's current device immediately.  The context 
 * will remain current to all of the threads that it was current to.  The next CUDA 
 * Runtime API call on any thread which requires an active context will trigger the 
 * reinitialization of that device's primary context.
 *
 * Note that primary contexts are shared resources. It is recommended that
 * the primary context not be reset except just before exit or to recover from an
 * unspecified launch failure.
 * 
 * \section CUDART_CUDA_context Context Interoperability
 *
 * Note that the use of multiple ::CUcontext s per device within a single process 
 * will substantially degrade performance and is strongly discouraged.  Instead,
 * it is highly recommended that the implicit one-to-one device-to-context mapping
 * for the process provided by the CUDA Runtime API be used.
 *
 * If a non-primary ::CUcontext created by the CUDA Driver API is current to a
 * thread then the CUDA Runtime API calls to that thread will operate on that 
 * ::CUcontext, with some exceptions listed below.  Interoperability between data
 * types is discussed in the following sections.
 *
 * The function ::cudaPointerGetAttributes() will return the error 
 * ::cudaErrorIncompatibleDriverContext if the pointer being queried was allocated by a 
 * non-primary context.  The function ::cudaDeviceEnablePeerAccess() and the rest of 
 * the peer access API may not be called when a non-primary ::CUcontext is current.  
 * To use the pointer query and peer access APIs with a context created using the 
 * CUDA Driver API, it is necessary that the CUDA Driver API be used to access
 * these features.
 *
 * All CUDA Runtime API state (e.g, global variables' addresses and values) travels
 * with its underlying ::CUcontext.  In particular, if a ::CUcontext is moved from one 
 * thread to another then all CUDA Runtime API state will move to that thread as well.
 *
 * Please note that attaching to legacy contexts (those with a version of 3010 as returned
 * by ::cuCtxGetApiVersion()) is not possible. The CUDA Runtime will return
 * ::cudaErrorIncompatibleDriverContext in such cases.
 *
 * \section CUDART_CUDA_stream Interactions between CUstream and cudaStream_t
 *
 * The types ::CUstream and ::cudaStream_t are identical and may be used interchangeably.
 *
 * \section CUDART_CUDA_event Interactions between CUevent and cudaEvent_t
 *
 * The types ::CUevent and ::cudaEvent_t are identical and may be used interchangeably.
 *
 * \section CUDART_CUDA_array Interactions between CUarray and cudaArray_t 
 *
 * The types ::CUarray and struct ::cudaArray * represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUarray in a CUDA Runtime API function which takes a struct ::cudaArray *,
 * it is necessary to explicitly cast the ::CUarray to a struct ::cudaArray *.
 *
 * In order to use a struct ::cudaArray * in a CUDA Driver API function which takes a ::CUarray,
 * it is necessary to explicitly cast the struct ::cudaArray * to a ::CUarray .
 *
 * \section CUDART_CUDA_graphicsResource Interactions between CUgraphicsResource and cudaGraphicsResource_t
 *
 * The types ::CUgraphicsResource and ::cudaGraphicsResource_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUgraphicsResource in a CUDA Runtime API function which takes a 
 * ::cudaGraphicsResource_t, it is necessary to explicitly cast the ::CUgraphicsResource 
 * to a ::cudaGraphicsResource_t.
 *
 * In order to use a ::cudaGraphicsResource_t in a CUDA Driver API function which takes a
 * ::CUgraphicsResource, it is necessary to explicitly cast the ::cudaGraphicsResource_t 
 * to a ::CUgraphicsResource.
 *
 * \section CUDART_CUDA_texture_objects Interactions between CUtexObject * and cudaTextureObject_t
 *
 * The types ::CUtexObject * and ::cudaTextureObject_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUtexObject * in a CUDA Runtime API function which takes a ::cudaTextureObject_t,
 * it is necessary to explicitly cast the ::CUtexObject * to a ::cudaTextureObject_t.
 *
 * In order to use a ::cudaTextureObject_t in a CUDA Driver API function which takes a ::CUtexObject *,
 * it is necessary to explicitly cast the ::cudaTextureObject_t to a ::CUtexObject *.
 *
 * \section CUDART_CUDA_surface_objects Interactions between CUsurfObject * and cudaSurfaceObject_t
 *
 * The types ::CUsurfObject * and ::cudaSurfaceObject_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUsurfObject * in a CUDA Runtime API function which takes a ::cudaSurfaceObjec_t,
 * it is necessary to explicitly cast the ::CUsurfObject * to a ::cudaSurfaceObject_t.
 *
 * In order to use a ::cudaSurfaceObject_t in a CUDA Driver API function which takes a ::CUsurfObject *,
 * it is necessary to explicitly cast the ::cudaSurfaceObject_t to a ::CUsurfObject *.
 *
 * \section CUDART_CUDA_module Interactions between CUfunction and cudaFunction_t
 *
 * The types ::CUfunction and ::cudaFunction_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::cudaFunction_t in a CUDA Driver API function which takes a ::CUfunction,
 * it is necessary to explicitly cast the ::cudaFunction_t to a ::CUfunction.
 *
 */

// extern __host__ cudaError_t CUDARTAPI_CDECL cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr) CUDART_NOT_IMPLEMENTED;

#if defined(__cplusplus)
}
#endif /* __cplusplus */
