#include "cuda_runtime_u.h"
#include <errno.h>

typedef struct ms_cudaLaunchKernelByName_t {
	cudaError_t ms_retval;
	char* ms_funcname;
	dim3 ms_gridDim;
	dim3 ms_blockDim;
	void* ms_argbuf;
	int ms_argbufsize;
	uint32_t* ms_parameters;
	int ms_partotal_size;
	size_t ms_sharedMem;
	cudaStream_t ms_stream;
} ms_cudaLaunchKernelByName_t;

typedef struct ms_cudaFuncGetParametersByName_t {
	cudaError_t ms_retval;
	uint32_t* ms_n_par;
	uint32_t* ms_parameters;
	char* ms_entryname;
	int ms_name_len;
} ms_cudaFuncGetParametersByName_t;

typedef struct ms_cudaThreadSynchronize_t {
	cudaError_t ms_retval;
} ms_cudaThreadSynchronize_t;

typedef struct ms_cudaDeviceSynchronize_t {
	cudaError_t ms_retval;
} ms_cudaDeviceSynchronize_t;

typedef struct ms_cudaGetLastError_t {
	cudaError_t ms_retval;
} ms_cudaGetLastError_t;

typedef struct ms_cudaGetDeviceCount_t {
	cudaError_t ms_retval;
	int* ms_count;
} ms_cudaGetDeviceCount_t;

typedef struct ms_cudaGetDeviceProperties_t {
	cudaError_t ms_retval;
	struct cudaDeviceProp* ms_prop;
	int ms_device;
} ms_cudaGetDeviceProperties_t;

typedef struct ms_cudaDeviceGetAttribute_t {
	cudaError_t ms_retval;
	int* ms_value;
	enum cudaDeviceAttr ms_attr;
	int ms_device;
} ms_cudaDeviceGetAttribute_t;

typedef struct ms_cudaDeviceGetDefaultMemPool_t {
	cudaError_t ms_retval;
	cudaMemPool_t* ms_memPool;
	int ms_device;
} ms_cudaDeviceGetDefaultMemPool_t;

typedef struct ms_cudaDeviceSetMemPool_t {
	cudaError_t ms_retval;
	int ms_device;
	cudaMemPool_t ms_memPool;
} ms_cudaDeviceSetMemPool_t;

typedef struct ms_cudaDeviceGetMemPool_t {
	cudaError_t ms_retval;
	cudaMemPool_t* ms_memPool;
	int ms_device;
} ms_cudaDeviceGetMemPool_t;

typedef struct ms_cudaChooseDevice_t {
	cudaError_t ms_retval;
	int* ms_device;
	struct cudaDeviceProp* ms_prop;
} ms_cudaChooseDevice_t;

typedef struct ms_cudaSetDevice_t {
	cudaError_t ms_retval;
	int ms_device;
} ms_cudaSetDevice_t;

typedef struct ms_cudaGetDevice_t {
	cudaError_t ms_retval;
	int* ms_device;
} ms_cudaGetDevice_t;

typedef struct ms_cudaSetValidDevices_t {
	cudaError_t ms_retval;
	int* ms_device_arr;
	int ms_len;
} ms_cudaSetValidDevices_t;

typedef struct ms_cudaSetDeviceFlags_t {
	cudaError_t ms_retval;
	unsigned int ms_flags;
} ms_cudaSetDeviceFlags_t;

typedef struct ms_cudaGetDeviceFlags_t {
	cudaError_t ms_retval;
	unsigned int* ms_flags;
} ms_cudaGetDeviceFlags_t;

typedef struct ms_cudaStreamCreate_t {
	cudaError_t ms_retval;
	cudaStream_t* ms_pStream;
} ms_cudaStreamCreate_t;

typedef struct ms_cudaStreamCreateWithFlags_t {
	cudaError_t ms_retval;
	cudaStream_t* ms_pStream;
	unsigned int ms_flags;
} ms_cudaStreamCreateWithFlags_t;

typedef struct ms_cudaStreamCreateWithPriority_t {
	cudaError_t ms_retval;
	cudaStream_t* ms_pStream;
	unsigned int ms_flags;
	int ms_priority;
} ms_cudaStreamCreateWithPriority_t;

typedef struct ms_cudaStreamGetPriority_t {
	cudaError_t ms_retval;
	cudaStream_t ms_hStream;
	int* ms_priority;
} ms_cudaStreamGetPriority_t;

typedef struct ms_cudaStreamGetFlags_t {
	cudaError_t ms_retval;
	cudaStream_t ms_hStream;
	unsigned int* ms_flags;
} ms_cudaStreamGetFlags_t;

typedef struct ms_cudaCtxResetPersistingL2Cache_t {
	cudaError_t ms_retval;
} ms_cudaCtxResetPersistingL2Cache_t;

typedef struct ms_cudaStreamCopyAttributes_t {
	cudaError_t ms_retval;
	cudaStream_t ms_dst;
	cudaStream_t ms_src;
} ms_cudaStreamCopyAttributes_t;

typedef struct ms_cudaStreamGetAttribute_t {
	cudaError_t ms_retval;
	cudaStream_t ms_hStream;
	enum cudaStreamAttrID ms_attr;
	union cudaStreamAttrValue* ms_value_out;
} ms_cudaStreamGetAttribute_t;

typedef struct ms_cudaStreamSetAttribute_t {
	cudaError_t ms_retval;
	cudaStream_t ms_hStream;
	enum cudaStreamAttrID ms_attr;
	union cudaStreamAttrValue* ms_value;
} ms_cudaStreamSetAttribute_t;

typedef struct ms_cudaStreamDestroy_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
} ms_cudaStreamDestroy_t;

typedef struct ms_cudaStreamWaitEvent_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
	cudaEvent_t ms_event;
	unsigned int ms_flags;
} ms_cudaStreamWaitEvent_t;

typedef struct ms_cudaStreamSynchronize_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
} ms_cudaStreamSynchronize_t;

typedef struct ms_cudaStreamQuery_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
} ms_cudaStreamQuery_t;

typedef struct ms_cudaStreamAttachMemAsync_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
	void* ms_devPtr;
	size_t ms_length;
	unsigned int ms_flags;
} ms_cudaStreamAttachMemAsync_t;

typedef struct ms_cudaStreamBeginCapture_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
	enum cudaStreamCaptureMode ms_mode;
} ms_cudaStreamBeginCapture_t;

typedef struct ms_cudaThreadExchangeStreamCaptureMode_t {
	cudaError_t ms_retval;
	enum cudaStreamCaptureMode* ms_mode;
} ms_cudaThreadExchangeStreamCaptureMode_t;

typedef struct ms_cudaStreamEndCapture_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
	cudaGraph_t* ms_pGraph;
} ms_cudaStreamEndCapture_t;

typedef struct ms_cudaStreamIsCapturing_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
	enum cudaStreamCaptureStatus* ms_pCaptureStatus;
} ms_cudaStreamIsCapturing_t;

typedef struct ms_cudaStreamGetCaptureInfo_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
	enum cudaStreamCaptureStatus* ms_pCaptureStatus;
	unsigned long long* ms_pId;
} ms_cudaStreamGetCaptureInfo_t;

typedef struct ms_cudaStreamUpdateCaptureDependencies_t {
	cudaError_t ms_retval;
	cudaStream_t ms_stream;
	cudaGraphNode_t* ms_dependencies;
	size_t ms_numDependencies;
	unsigned int ms_flags;
} ms_cudaStreamUpdateCaptureDependencies_t;

typedef struct ms_cudaMallocManaged_t {
	cudaError_t ms_retval;
	void** ms_devPtr;
	size_t ms_size;
	unsigned int ms_flags;
} ms_cudaMallocManaged_t;

typedef struct ms_cudaMalloc_t {
	cudaError_t ms_retval;
	void** ms_devPtr;
	size_t ms_size;
} ms_cudaMalloc_t;

typedef struct ms_cudaMallocHost_t {
	cudaError_t ms_retval;
	void** ms_ptr;
	size_t ms_size;
} ms_cudaMallocHost_t;

typedef struct ms_cudaMallocPitch_t {
	cudaError_t ms_retval;
	void** ms_devPtr;
	size_t* ms_pitch;
	size_t ms_width;
	size_t ms_height;
} ms_cudaMallocPitch_t;

typedef struct ms_cudaMallocArray_t {
	cudaError_t ms_retval;
	cudaArray_t* ms_array;
	struct cudaChannelFormatDesc* ms_desc;
	size_t ms_width;
	size_t ms_height;
	unsigned int ms_flags;
} ms_cudaMallocArray_t;

typedef struct ms_cudaFree_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
} ms_cudaFree_t;

typedef struct ms_cudaFreeHost_t {
	cudaError_t ms_retval;
	void* ms_ptr;
} ms_cudaFreeHost_t;

typedef struct ms_cudaFreeArray_t {
	cudaError_t ms_retval;
	cudaArray_t ms_array;
} ms_cudaFreeArray_t;

typedef struct ms_cudaFreeMipmappedArray_t {
	cudaError_t ms_retval;
	cudaMipmappedArray_t ms_mipmappedArray;
} ms_cudaFreeMipmappedArray_t;

typedef struct ms_cudaHostAlloc_t {
	cudaError_t ms_retval;
	void** ms_pHost;
	size_t ms_size;
	unsigned int ms_flags;
} ms_cudaHostAlloc_t;

typedef struct ms_cudaHostRegister_t {
	cudaError_t ms_retval;
	void* ms_ptr;
	size_t ms_size;
	unsigned int ms_flags;
} ms_cudaHostRegister_t;

typedef struct ms_cudaHostUnregister_t {
	cudaError_t ms_retval;
	void* ms_ptr;
} ms_cudaHostUnregister_t;

typedef struct ms_cudaHostGetDevicePointer_t {
	cudaError_t ms_retval;
	void** ms_pDevice;
	void* ms_pHost;
	unsigned int ms_flags;
} ms_cudaHostGetDevicePointer_t;

typedef struct ms_cudaHostGetFlags_t {
	cudaError_t ms_retval;
	unsigned int* ms_pFlags;
	void* ms_pHost;
} ms_cudaHostGetFlags_t;

typedef struct ms_cudaMalloc3D_t {
	cudaError_t ms_retval;
	struct cudaPitchedPtr* ms_pitchedDevPtr;
	struct cudaExtent ms_extent;
} ms_cudaMalloc3D_t;

typedef struct ms_cudaMalloc3DArray_t {
	cudaError_t ms_retval;
	cudaArray_t* ms_array;
	struct cudaChannelFormatDesc* ms_desc;
	struct cudaExtent ms_extent;
	unsigned int ms_flags;
} ms_cudaMalloc3DArray_t;

typedef struct ms_cudaMallocMipmappedArray_t {
	cudaError_t ms_retval;
	cudaMipmappedArray_t* ms_mipmappedArray;
	struct cudaChannelFormatDesc* ms_desc;
	struct cudaExtent ms_extent;
	unsigned int ms_numLevels;
	unsigned int ms_flags;
} ms_cudaMallocMipmappedArray_t;

typedef struct ms_cudaGetMipmappedArrayLevel_t {
	cudaError_t ms_retval;
	cudaArray_t* ms_levelArray;
	cudaMipmappedArray_const_t ms_mipmappedArray;
	unsigned int ms_level;
} ms_cudaGetMipmappedArrayLevel_t;

typedef struct ms_cudaMemcpy3D_t {
	cudaError_t ms_retval;
	struct cudaMemcpy3DParms* ms_p;
} ms_cudaMemcpy3D_t;

typedef struct ms_cudaMemcpy3DPeer_t {
	cudaError_t ms_retval;
	struct cudaMemcpy3DPeerParms* ms_p;
} ms_cudaMemcpy3DPeer_t;

typedef struct ms_cudaMemcpy3DAsync_t {
	cudaError_t ms_retval;
	struct cudaMemcpy3DParms* ms_p;
	cudaStream_t ms_stream;
} ms_cudaMemcpy3DAsync_t;

typedef struct ms_cudaMemcpy3DPeerAsync_t {
	cudaError_t ms_retval;
	struct cudaMemcpy3DPeerParms* ms_p;
	cudaStream_t ms_stream;
} ms_cudaMemcpy3DPeerAsync_t;

typedef struct ms_cudaMemGetInfo_t {
	cudaError_t ms_retval;
	size_t* ms_free;
	size_t* ms_total;
} ms_cudaMemGetInfo_t;

typedef struct ms_cudaArrayGetInfo_t {
	cudaError_t ms_retval;
	struct cudaChannelFormatDesc* ms_desc;
	struct cudaExtent* ms_extent;
	unsigned int* ms_flags;
	cudaArray_t ms_array;
} ms_cudaArrayGetInfo_t;

typedef struct ms_cudaArrayGetPlane_t {
	cudaError_t ms_retval;
	cudaArray_t* ms_pPlaneArray;
	cudaArray_t ms_hArray;
	unsigned int ms_planeIdx;
} ms_cudaArrayGetPlane_t;

typedef struct ms_cudaArrayGetSparseProperties_t {
	cudaError_t ms_retval;
	struct cudaArraySparseProperties* ms_sparseProperties;
	cudaArray_t ms_array;
} ms_cudaArrayGetSparseProperties_t;

typedef struct ms_cudaMipmappedArrayGetSparseProperties_t {
	cudaError_t ms_retval;
	struct cudaArraySparseProperties* ms_sparseProperties;
	cudaMipmappedArray_t ms_mipmap;
} ms_cudaMipmappedArrayGetSparseProperties_t;

typedef struct ms_cudaMemcpyNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyNone_t;

typedef struct ms_cudaMemcpySrc_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpySrc_t;

typedef struct ms_cudaMemcpyDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyDst_t;

typedef struct ms_cudaMemcpySrcDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpySrcDst_t;

typedef struct ms_cudaMemcpyPeer_t {
	cudaError_t ms_retval;
	void* ms_dst;
	int ms_dstDevice;
	void* ms_src;
	int ms_srcDevice;
	size_t ms_count;
} ms_cudaMemcpyPeer_t;

typedef struct ms_cudaMemcpy2DNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DNone_t;

typedef struct ms_cudaMemcpy2DSrc_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DSrc_t;

typedef struct ms_cudaMemcpy2DDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DDst_t;

typedef struct ms_cudaMemcpy2DSrcDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DSrcDst_t;

typedef struct ms_cudaMemcpy2DToArrayNone_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DToArrayNone_t;

typedef struct ms_cudaMemcpy2DToArraySrc_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DToArraySrc_t;

typedef struct ms_cudaMemcpy2DFromArrayNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DFromArrayNone_t;

typedef struct ms_cudaMemcpy2DFromArrayDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DFromArrayDst_t;

typedef struct ms_cudaMemcpy2DArrayToArray_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffsetDst;
	size_t ms_hOffsetDst;
	cudaArray_const_t ms_src;
	size_t ms_wOffsetSrc;
	size_t ms_hOffsetSrc;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpy2DArrayToArray_t;

typedef struct ms_cudaMemcpyToSymbolNone_t {
	cudaError_t ms_retval;
	void* ms_symbol;
	void* ms_src;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyToSymbolNone_t;

typedef struct ms_cudaMemcpyToSymbolSrc_t {
	cudaError_t ms_retval;
	void* ms_symbol;
	void* ms_src;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyToSymbolSrc_t;

typedef struct ms_cudaMemcpyFromSymbolNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_symbol;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyFromSymbolNone_t;

typedef struct ms_cudaMemcpyFromSymbolDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_symbol;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyFromSymbolDst_t;

typedef struct ms_cudaMemcpyAsyncNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyAsyncNone_t;

typedef struct ms_cudaMemcpyAsyncSrc_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyAsyncSrc_t;

typedef struct ms_cudaMemcpyAsyncDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyAsyncDst_t;

typedef struct ms_cudaMemcpyAsyncSrcDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyAsyncSrcDst_t;

typedef struct ms_cudaMemcpyPeerAsync_t {
	cudaError_t ms_retval;
	void* ms_dst;
	int ms_dstDevice;
	void* ms_src;
	int ms_srcDevice;
	size_t ms_count;
	cudaStream_t ms_stream;
} ms_cudaMemcpyPeerAsync_t;

typedef struct ms_cudaMemcpy2DAsyncNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DAsyncNone_t;

typedef struct ms_cudaMemcpy2DAsyncSrc_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DAsyncSrc_t;

typedef struct ms_cudaMemcpy2DAsyncDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DAsyncDst_t;

typedef struct ms_cudaMemcpy2DAsyncSrcDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DAsyncSrcDst_t;

typedef struct ms_cudaMemcpy2DToArrayAsyncNone_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DToArrayAsyncNone_t;

typedef struct ms_cudaMemcpy2DToArrayAsyncSrc_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_spitch;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DToArrayAsyncSrc_t;

typedef struct ms_cudaMemcpy2DFromArrayAsyncNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DFromArrayAsyncNone_t;

typedef struct ms_cudaMemcpy2DFromArrayAsyncDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	size_t ms_dpitch;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_width;
	size_t ms_height;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpy2DFromArrayAsyncDst_t;

typedef struct ms_cudaMemcpyToSymbolAsyncNone_t {
	cudaError_t ms_retval;
	void* ms_symbol;
	void* ms_src;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyToSymbolAsyncNone_t;

typedef struct ms_cudaMemcpyToSymbolAsyncSrc_t {
	cudaError_t ms_retval;
	void* ms_symbol;
	void* ms_src;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyToSymbolAsyncSrc_t;

typedef struct ms_cudaMemcpyFromSymbolAsyncNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_symbol;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyFromSymbolAsyncNone_t;

typedef struct ms_cudaMemcpyFromSymbolAsyncDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	void* ms_symbol;
	size_t ms_count;
	size_t ms_offset;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyFromSymbolAsyncDst_t;

typedef struct ms_cudaMemset_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
	int ms_value;
	size_t ms_count;
} ms_cudaMemset_t;

typedef struct ms_cudaMemset2D_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
	size_t ms_pitch;
	int ms_value;
	size_t ms_width;
	size_t ms_height;
} ms_cudaMemset2D_t;

typedef struct ms_cudaMemset3D_t {
	cudaError_t ms_retval;
	struct cudaPitchedPtr ms_pitchedDevPtr;
	int ms_value;
	struct cudaExtent ms_extent;
} ms_cudaMemset3D_t;

typedef struct ms_cudaMemsetAsync_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
	int ms_value;
	size_t ms_count;
	cudaStream_t ms_stream;
} ms_cudaMemsetAsync_t;

typedef struct ms_cudaMemset2DAsync_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
	size_t ms_pitch;
	int ms_value;
	size_t ms_width;
	size_t ms_height;
	cudaStream_t ms_stream;
} ms_cudaMemset2DAsync_t;

typedef struct ms_cudaMemset3DAsync_t {
	cudaError_t ms_retval;
	struct cudaPitchedPtr ms_pitchedDevPtr;
	int ms_value;
	struct cudaExtent ms_extent;
	cudaStream_t ms_stream;
} ms_cudaMemset3DAsync_t;

typedef struct ms_cudaGetSymbolAddress_t {
	cudaError_t ms_retval;
	void** ms_devPtr;
	void* ms_symbol;
} ms_cudaGetSymbolAddress_t;

typedef struct ms_cudaGetSymbolSize_t {
	cudaError_t ms_retval;
	size_t* ms_size;
	void* ms_symbol;
} ms_cudaGetSymbolSize_t;

typedef struct ms_cudaMemPrefetchAsync_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
	size_t ms_count;
	int ms_dstDevice;
	cudaStream_t ms_stream;
} ms_cudaMemPrefetchAsync_t;

typedef struct ms_cudaMemAdvise_t {
	cudaError_t ms_retval;
	void* ms_devPtr;
	size_t ms_count;
	enum cudaMemoryAdvise ms_advice;
	int ms_device;
} ms_cudaMemAdvise_t;

typedef struct ms_cudaMemRangeGetAttribute_t {
	cudaError_t ms_retval;
	void* ms_data;
	size_t ms_dataSize;
	enum cudaMemRangeAttribute ms_attribute;
	void* ms_devPtr;
	size_t ms_count;
} ms_cudaMemRangeGetAttribute_t;

typedef struct ms_cudaMemcpyToArrayNone_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyToArrayNone_t;

typedef struct ms_cudaMemcpyToArraySrc_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyToArraySrc_t;

typedef struct ms_cudaMemcpyFromArrayNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyFromArrayNone_t;

typedef struct ms_cudaMemcpyFromArrayDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyFromArrayDst_t;

typedef struct ms_cudaMemcpyArrayToArray_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffsetDst;
	size_t ms_hOffsetDst;
	cudaArray_const_t ms_src;
	size_t ms_wOffsetSrc;
	size_t ms_hOffsetSrc;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
} ms_cudaMemcpyArrayToArray_t;

typedef struct ms_cudaMemcpyToArrayAsyncNone_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyToArrayAsyncNone_t;

typedef struct ms_cudaMemcpyToArrayAsyncSrc_t {
	cudaError_t ms_retval;
	cudaArray_t ms_dst;
	size_t ms_wOffset;
	size_t ms_hOffset;
	void* ms_src;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyToArrayAsyncSrc_t;

typedef struct ms_cudaMemcpyFromArrayAsyncNone_t {
	cudaError_t ms_retval;
	void* ms_dst;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyFromArrayAsyncNone_t;

typedef struct ms_cudaMemcpyFromArrayAsyncDst_t {
	cudaError_t ms_retval;
	void* ms_dst;
	cudaArray_const_t ms_src;
	size_t ms_wOffset;
	size_t ms_hOffset;
	size_t ms_count;
	enum cudaMemcpyKind ms_kind;
	cudaStream_t ms_stream;
} ms_cudaMemcpyFromArrayAsyncDst_t;

cudaError_t cudaLaunchKernelByName(char* funcname, dim3 gridDim, dim3 blockDim, void* argbuf, int argbufsize, uint32_t* parameters, int partotal_size, size_t sharedMem, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaLaunchKernelByName_t* ms = TEE_CAST(ms_cudaLaunchKernelByName_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaLaunchKernelByName_t), funcname, strlen((const char*)funcname) + 1);
	memcpy(enclave_buffer + sizeof(ms_cudaLaunchKernelByName_t) + strlen((const char*)funcname) + 1, argbuf, argbufsize);
	memcpy(enclave_buffer + sizeof(ms_cudaLaunchKernelByName_t) + strlen((const char*)funcname) + 1 + argbufsize, parameters, partotal_size);

	ms->ms_funcname = funcname;
	ms->ms_gridDim = gridDim;
	ms->ms_blockDim = blockDim;
	ms->ms_argbuf = argbuf;
	ms->ms_argbufsize = argbufsize;
	ms->ms_parameters = parameters;
	ms->ms_partotal_size = partotal_size;
	ms->ms_sharedMem = sharedMem;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaLaunchKernelByName_t) + strlen((const char*)funcname) + 1 + argbufsize + partotal_size;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(0, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaFuncGetParametersByName(uint32_t* n_par, uint32_t* parameters, const char* entryname, int name_len)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaFuncGetParametersByName_t* ms = TEE_CAST(ms_cudaFuncGetParametersByName_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaFuncGetParametersByName_t) + 4 + 80, entryname, name_len);

	ms->ms_n_par = n_par;
	ms->ms_parameters = parameters;
	ms->ms_entryname = (char*)entryname;
	ms->ms_name_len = name_len;
	bufsize = sizeof(ms_cudaFuncGetParametersByName_t) + 4 + 80 + name_len;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(1, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(n_par, enclave_buffer + sizeof(ms_cudaFuncGetParametersByName_t), 4);
		memcpy(parameters, enclave_buffer + sizeof(ms_cudaFuncGetParametersByName_t) + 4, 80);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaThreadSynchronize()
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaThreadSynchronize_t* ms = TEE_CAST(ms_cudaThreadSynchronize_t*, enclave_buffer);;
	

	bufsize = sizeof(ms_cudaThreadSynchronize_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(2, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaDeviceSynchronize()
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaDeviceSynchronize_t* ms = TEE_CAST(ms_cudaDeviceSynchronize_t*, enclave_buffer);;
	

	bufsize = sizeof(ms_cudaDeviceSynchronize_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(3, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetLastError()
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetLastError_t* ms = TEE_CAST(ms_cudaGetLastError_t*, enclave_buffer);;
	

	bufsize = sizeof(ms_cudaGetLastError_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(4, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetDeviceCount(int* count)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetDeviceCount_t* ms = TEE_CAST(ms_cudaGetDeviceCount_t*, enclave_buffer);;
	

	ms->ms_count = count;
	bufsize = sizeof(ms_cudaGetDeviceCount_t) + 1 * sizeof(*count);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(5, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(count, enclave_buffer + sizeof(ms_cudaGetDeviceCount_t), 1 * sizeof(*count));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetDeviceProperties_t* ms = TEE_CAST(ms_cudaGetDeviceProperties_t*, enclave_buffer);;
	

	ms->ms_prop = prop;
	ms->ms_device = device;
	bufsize = sizeof(ms_cudaGetDeviceProperties_t) + 1 * sizeof(*prop);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(6, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(prop, enclave_buffer + sizeof(ms_cudaGetDeviceProperties_t), 1 * sizeof(*prop));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaDeviceGetAttribute_t* ms = TEE_CAST(ms_cudaDeviceGetAttribute_t*, enclave_buffer);;
	

	ms->ms_value = value;
	ms->ms_attr = attr;
	ms->ms_device = device;
	bufsize = sizeof(ms_cudaDeviceGetAttribute_t) + 1 * sizeof(*value);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(7, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(value, enclave_buffer + sizeof(ms_cudaDeviceGetAttribute_t), 1 * sizeof(*value));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaDeviceGetDefaultMemPool_t* ms = TEE_CAST(ms_cudaDeviceGetDefaultMemPool_t*, enclave_buffer);;
	

	ms->ms_memPool = memPool;
	ms->ms_device = device;
	bufsize = sizeof(ms_cudaDeviceGetDefaultMemPool_t) + 1 * sizeof(*memPool);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(8, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(memPool, enclave_buffer + sizeof(ms_cudaDeviceGetDefaultMemPool_t), 1 * sizeof(*memPool));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaDeviceSetMemPool_t* ms = TEE_CAST(ms_cudaDeviceSetMemPool_t*, enclave_buffer);;
	

	ms->ms_device = device;
	ms->ms_memPool = memPool;
	bufsize = sizeof(ms_cudaDeviceSetMemPool_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(9, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaDeviceGetMemPool_t* ms = TEE_CAST(ms_cudaDeviceGetMemPool_t*, enclave_buffer);;
	

	ms->ms_memPool = memPool;
	ms->ms_device = device;
	bufsize = sizeof(ms_cudaDeviceGetMemPool_t) + 1 * sizeof(*memPool);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(10, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(memPool, enclave_buffer + sizeof(ms_cudaDeviceGetMemPool_t), 1 * sizeof(*memPool));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaChooseDevice_t* ms = TEE_CAST(ms_cudaChooseDevice_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaChooseDevice_t) + 1 * sizeof(*device), prop, 1 * sizeof(*prop));

	ms->ms_device = device;
	ms->ms_prop = (struct cudaDeviceProp*)prop;
	bufsize = sizeof(ms_cudaChooseDevice_t) + 1 * sizeof(*device) + 1 * sizeof(*prop);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(11, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(device, enclave_buffer + sizeof(ms_cudaChooseDevice_t), 1 * sizeof(*device));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaSetDevice(int device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaSetDevice_t* ms = TEE_CAST(ms_cudaSetDevice_t*, enclave_buffer);;
	

	ms->ms_device = device;
	bufsize = sizeof(ms_cudaSetDevice_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(12, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetDevice(int* device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetDevice_t* ms = TEE_CAST(ms_cudaGetDevice_t*, enclave_buffer);;
	

	ms->ms_device = device;
	bufsize = sizeof(ms_cudaGetDevice_t) + 1 * sizeof(*device);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(13, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(device, enclave_buffer + sizeof(ms_cudaGetDevice_t), 1 * sizeof(*device));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaSetValidDevices(int* device_arr, int len)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaSetValidDevices_t* ms = TEE_CAST(ms_cudaSetValidDevices_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaSetValidDevices_t), device_arr, len * sizeof(*device_arr));

	ms->ms_device_arr = device_arr;
	ms->ms_len = len;
	bufsize = sizeof(ms_cudaSetValidDevices_t) + len * sizeof(*device_arr);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(14, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaSetDeviceFlags_t* ms = TEE_CAST(ms_cudaSetDeviceFlags_t*, enclave_buffer);;
	

	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaSetDeviceFlags_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(15, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetDeviceFlags_t* ms = TEE_CAST(ms_cudaGetDeviceFlags_t*, enclave_buffer);;
	

	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaGetDeviceFlags_t) + 1 * sizeof(*flags);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(16, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(flags, enclave_buffer + sizeof(ms_cudaGetDeviceFlags_t), 1 * sizeof(*flags));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamCreate_t* ms = TEE_CAST(ms_cudaStreamCreate_t*, enclave_buffer);;
	

	ms->ms_pStream = pStream;
	bufsize = sizeof(ms_cudaStreamCreate_t) + 1 * sizeof(*pStream);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(17, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pStream, enclave_buffer + sizeof(ms_cudaStreamCreate_t), 1 * sizeof(*pStream));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamCreateWithFlags_t* ms = TEE_CAST(ms_cudaStreamCreateWithFlags_t*, enclave_buffer);;
	

	ms->ms_pStream = pStream;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaStreamCreateWithFlags_t) + 1 * sizeof(*pStream);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(18, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pStream, enclave_buffer + sizeof(ms_cudaStreamCreateWithFlags_t), 1 * sizeof(*pStream));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamCreateWithPriority_t* ms = TEE_CAST(ms_cudaStreamCreateWithPriority_t*, enclave_buffer);;
	

	ms->ms_pStream = pStream;
	ms->ms_flags = flags;
	ms->ms_priority = priority;
	bufsize = sizeof(ms_cudaStreamCreateWithPriority_t) + 1 * sizeof(*pStream);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(19, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pStream, enclave_buffer + sizeof(ms_cudaStreamCreateWithPriority_t), 1 * sizeof(*pStream));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamGetPriority_t* ms = TEE_CAST(ms_cudaStreamGetPriority_t*, enclave_buffer);;
	

	ms->ms_hStream = hStream;
	ms->ms_priority = priority;
	bufsize = sizeof(ms_cudaStreamGetPriority_t) + 1 * sizeof(*priority);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(20, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(priority, enclave_buffer + sizeof(ms_cudaStreamGetPriority_t), 1 * sizeof(*priority));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamGetFlags_t* ms = TEE_CAST(ms_cudaStreamGetFlags_t*, enclave_buffer);;
	

	ms->ms_hStream = hStream;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaStreamGetFlags_t) + 1 * sizeof(*flags);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(21, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(flags, enclave_buffer + sizeof(ms_cudaStreamGetFlags_t), 1 * sizeof(*flags));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaCtxResetPersistingL2Cache()
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaCtxResetPersistingL2Cache_t* ms = TEE_CAST(ms_cudaCtxResetPersistingL2Cache_t*, enclave_buffer);;
	

	bufsize = sizeof(ms_cudaCtxResetPersistingL2Cache_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(22, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamCopyAttributes_t* ms = TEE_CAST(ms_cudaStreamCopyAttributes_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = src;
	bufsize = sizeof(ms_cudaStreamCopyAttributes_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(23, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr, union cudaStreamAttrValue* value_out)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamGetAttribute_t* ms = TEE_CAST(ms_cudaStreamGetAttribute_t*, enclave_buffer);;
	

	ms->ms_hStream = hStream;
	ms->ms_attr = attr;
	ms->ms_value_out = value_out;
	bufsize = sizeof(ms_cudaStreamGetAttribute_t) + 1 * sizeof(*value_out);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(24, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(value_out, enclave_buffer + sizeof(ms_cudaStreamGetAttribute_t), 1 * sizeof(*value_out));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr, const union cudaStreamAttrValue* value)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamSetAttribute_t* ms = TEE_CAST(ms_cudaStreamSetAttribute_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaStreamSetAttribute_t), value, 1 * sizeof(*value));

	ms->ms_hStream = hStream;
	ms->ms_attr = attr;
	ms->ms_value = (union cudaStreamAttrValue*)value;
	bufsize = sizeof(ms_cudaStreamSetAttribute_t) + 1 * sizeof(*value);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(25, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamDestroy_t* ms = TEE_CAST(ms_cudaStreamDestroy_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaStreamDestroy_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(26, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamWaitEvent_t* ms = TEE_CAST(ms_cudaStreamWaitEvent_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	ms->ms_event = event;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaStreamWaitEvent_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(27, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamSynchronize_t* ms = TEE_CAST(ms_cudaStreamSynchronize_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaStreamSynchronize_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(28, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamQuery_t* ms = TEE_CAST(ms_cudaStreamQuery_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaStreamQuery_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(29, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamAttachMemAsync_t* ms = TEE_CAST(ms_cudaStreamAttachMemAsync_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	ms->ms_devPtr = devPtr;
	ms->ms_length = length;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaStreamAttachMemAsync_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(30, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamBeginCapture_t* ms = TEE_CAST(ms_cudaStreamBeginCapture_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	ms->ms_mode = mode;
	bufsize = sizeof(ms_cudaStreamBeginCapture_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(31, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaThreadExchangeStreamCaptureMode_t* ms = TEE_CAST(ms_cudaThreadExchangeStreamCaptureMode_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaThreadExchangeStreamCaptureMode_t), mode, 1 * sizeof(*mode));

	ms->ms_mode = mode;
	bufsize = sizeof(ms_cudaThreadExchangeStreamCaptureMode_t) + 1 * sizeof(*mode);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(32, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(mode, enclave_buffer + sizeof(ms_cudaThreadExchangeStreamCaptureMode_t), 1 * sizeof(*mode));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamEndCapture_t* ms = TEE_CAST(ms_cudaStreamEndCapture_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	ms->ms_pGraph = pGraph;
	bufsize = sizeof(ms_cudaStreamEndCapture_t) + 1 * sizeof(*pGraph);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(33, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pGraph, enclave_buffer + sizeof(ms_cudaStreamEndCapture_t), 1 * sizeof(*pGraph));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamIsCapturing_t* ms = TEE_CAST(ms_cudaStreamIsCapturing_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	ms->ms_pCaptureStatus = pCaptureStatus;
	bufsize = sizeof(ms_cudaStreamIsCapturing_t) + 1 * sizeof(*pCaptureStatus);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(34, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pCaptureStatus, enclave_buffer + sizeof(ms_cudaStreamIsCapturing_t), 1 * sizeof(*pCaptureStatus));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamGetCaptureInfo_t* ms = TEE_CAST(ms_cudaStreamGetCaptureInfo_t*, enclave_buffer);;
	

	ms->ms_stream = stream;
	ms->ms_pCaptureStatus = pCaptureStatus;
	ms->ms_pId = pId;
	bufsize = sizeof(ms_cudaStreamGetCaptureInfo_t) + 1 * sizeof(*pCaptureStatus) + 1 * sizeof(*pId);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(35, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pCaptureStatus, enclave_buffer + sizeof(ms_cudaStreamGetCaptureInfo_t), 1 * sizeof(*pCaptureStatus));
		memcpy(pId, enclave_buffer + sizeof(ms_cudaStreamGetCaptureInfo_t) + 1 * sizeof(*pCaptureStatus), 1 * sizeof(*pId));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaStreamUpdateCaptureDependencies_t* ms = TEE_CAST(ms_cudaStreamUpdateCaptureDependencies_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaStreamUpdateCaptureDependencies_t), dependencies, numDependencies * sizeof(*dependencies));

	ms->ms_stream = stream;
	ms->ms_dependencies = dependencies;
	ms->ms_numDependencies = numDependencies;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaStreamUpdateCaptureDependencies_t) + numDependencies * sizeof(*dependencies);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(36, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMallocManaged_t* ms = TEE_CAST(ms_cudaMallocManaged_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_size = size;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaMallocManaged_t) + 1 * sizeof(*devPtr);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(37, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(devPtr, enclave_buffer + sizeof(ms_cudaMallocManaged_t), 1 * sizeof(*devPtr));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMalloc_t* ms = TEE_CAST(ms_cudaMalloc_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_size = size;
	bufsize = sizeof(ms_cudaMalloc_t) + 1 * sizeof(*devPtr);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(38, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(devPtr, enclave_buffer + sizeof(ms_cudaMalloc_t), 1 * sizeof(*devPtr));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMallocHost(void** ptr, size_t size)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMallocHost_t* ms = TEE_CAST(ms_cudaMallocHost_t*, enclave_buffer);;
	

	ms->ms_ptr = ptr;
	ms->ms_size = size;
	bufsize = sizeof(ms_cudaMallocHost_t) + 1 * sizeof(*ptr);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(39, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(ptr, enclave_buffer + sizeof(ms_cudaMallocHost_t), 1 * sizeof(*ptr));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMallocPitch_t* ms = TEE_CAST(ms_cudaMallocPitch_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_pitch = pitch;
	ms->ms_width = width;
	ms->ms_height = height;
	bufsize = sizeof(ms_cudaMallocPitch_t) + 1 * sizeof(*devPtr) + 1 * sizeof(*pitch);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(40, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(devPtr, enclave_buffer + sizeof(ms_cudaMallocPitch_t), 1 * sizeof(*devPtr));
		memcpy(pitch, enclave_buffer + sizeof(ms_cudaMallocPitch_t) + 1 * sizeof(*devPtr), 1 * sizeof(*pitch));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMallocArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMallocArray_t* ms = TEE_CAST(ms_cudaMallocArray_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMallocArray_t) + 1 * sizeof(*array), desc, 1 * sizeof(*desc));

	ms->ms_array = array;
	ms->ms_desc = (struct cudaChannelFormatDesc*)desc;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaMallocArray_t) + 1 * sizeof(*array) + 1 * sizeof(*desc);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(41, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(array, enclave_buffer + sizeof(ms_cudaMallocArray_t), 1 * sizeof(*array));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaFree(void* devPtr)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaFree_t* ms = TEE_CAST(ms_cudaFree_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	bufsize = sizeof(ms_cudaFree_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(42, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaFreeHost(void* ptr)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaFreeHost_t* ms = TEE_CAST(ms_cudaFreeHost_t*, enclave_buffer);;
	

	ms->ms_ptr = ptr;
	bufsize = sizeof(ms_cudaFreeHost_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(43, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaFreeArray(cudaArray_t array)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaFreeArray_t* ms = TEE_CAST(ms_cudaFreeArray_t*, enclave_buffer);;
	

	ms->ms_array = array;
	bufsize = sizeof(ms_cudaFreeArray_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(44, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaFreeMipmappedArray_t* ms = TEE_CAST(ms_cudaFreeMipmappedArray_t*, enclave_buffer);;
	

	ms->ms_mipmappedArray = mipmappedArray;
	bufsize = sizeof(ms_cudaFreeMipmappedArray_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(45, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaHostAlloc_t* ms = TEE_CAST(ms_cudaHostAlloc_t*, enclave_buffer);;
	

	ms->ms_pHost = pHost;
	ms->ms_size = size;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaHostAlloc_t) + 1 * sizeof(*pHost);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(46, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pHost, enclave_buffer + sizeof(ms_cudaHostAlloc_t), 1 * sizeof(*pHost));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaHostRegister_t* ms = TEE_CAST(ms_cudaHostRegister_t*, enclave_buffer);;
	

	ms->ms_ptr = ptr;
	ms->ms_size = size;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaHostRegister_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(47, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaHostUnregister(void* ptr)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaHostUnregister_t* ms = TEE_CAST(ms_cudaHostUnregister_t*, enclave_buffer);;
	

	ms->ms_ptr = ptr;
	bufsize = sizeof(ms_cudaHostUnregister_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(48, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaHostGetDevicePointer_t* ms = TEE_CAST(ms_cudaHostGetDevicePointer_t*, enclave_buffer);;
	

	ms->ms_pDevice = pDevice;
	ms->ms_pHost = pHost;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaHostGetDevicePointer_t) + 1 * sizeof(*pDevice) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(49, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pDevice, enclave_buffer + sizeof(ms_cudaHostGetDevicePointer_t), 1 * sizeof(*pDevice));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaHostGetFlags_t* ms = TEE_CAST(ms_cudaHostGetFlags_t*, enclave_buffer);;
	

	ms->ms_pFlags = pFlags;
	ms->ms_pHost = pHost;
	bufsize = sizeof(ms_cudaHostGetFlags_t) + 1 * sizeof(*pFlags) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(50, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pFlags, enclave_buffer + sizeof(ms_cudaHostGetFlags_t), 1 * sizeof(*pFlags));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMalloc3D_t* ms = TEE_CAST(ms_cudaMalloc3D_t*, enclave_buffer);;
	

	ms->ms_pitchedDevPtr = pitchedDevPtr;
	ms->ms_extent = extent;
	bufsize = sizeof(ms_cudaMalloc3D_t) + 1 * sizeof(*pitchedDevPtr);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(51, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pitchedDevPtr, enclave_buffer + sizeof(ms_cudaMalloc3D_t), 1 * sizeof(*pitchedDevPtr));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMalloc3DArray_t* ms = TEE_CAST(ms_cudaMalloc3DArray_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMalloc3DArray_t) + 1 * sizeof(*array), desc, 1 * sizeof(*desc));

	ms->ms_array = array;
	ms->ms_desc = (struct cudaChannelFormatDesc*)desc;
	ms->ms_extent = extent;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaMalloc3DArray_t) + 1 * sizeof(*array) + 1 * sizeof(*desc);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(52, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(array, enclave_buffer + sizeof(ms_cudaMalloc3DArray_t), 1 * sizeof(*array));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMallocMipmappedArray_t* ms = TEE_CAST(ms_cudaMallocMipmappedArray_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMallocMipmappedArray_t) + 1 * sizeof(*mipmappedArray), desc, 1 * sizeof(*desc));

	ms->ms_mipmappedArray = mipmappedArray;
	ms->ms_desc = (struct cudaChannelFormatDesc*)desc;
	ms->ms_extent = extent;
	ms->ms_numLevels = numLevels;
	ms->ms_flags = flags;
	bufsize = sizeof(ms_cudaMallocMipmappedArray_t) + 1 * sizeof(*mipmappedArray) + 1 * sizeof(*desc);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(53, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(mipmappedArray, enclave_buffer + sizeof(ms_cudaMallocMipmappedArray_t), 1 * sizeof(*mipmappedArray));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetMipmappedArrayLevel_t* ms = TEE_CAST(ms_cudaGetMipmappedArrayLevel_t*, enclave_buffer);;
	

	ms->ms_levelArray = levelArray;
	ms->ms_mipmappedArray = mipmappedArray;
	ms->ms_level = level;
	bufsize = sizeof(ms_cudaGetMipmappedArrayLevel_t) + 1 * sizeof(*levelArray);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(54, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(levelArray, enclave_buffer + sizeof(ms_cudaGetMipmappedArrayLevel_t), 1 * sizeof(*levelArray));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy3D_t* ms = TEE_CAST(ms_cudaMemcpy3D_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy3D_t), p, 1 * sizeof(*p));

	ms->ms_p = (struct cudaMemcpy3DParms*)p;
	bufsize = sizeof(ms_cudaMemcpy3D_t) + 1 * sizeof(*p);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(55, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms* p)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy3DPeer_t* ms = TEE_CAST(ms_cudaMemcpy3DPeer_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy3DPeer_t), p, 1 * sizeof(*p));

	ms->ms_p = (struct cudaMemcpy3DPeerParms*)p;
	bufsize = sizeof(ms_cudaMemcpy3DPeer_t) + 1 * sizeof(*p);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(56, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy3DAsync_t* ms = TEE_CAST(ms_cudaMemcpy3DAsync_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy3DAsync_t), p, 1 * sizeof(*p));

	ms->ms_p = (struct cudaMemcpy3DParms*)p;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy3DAsync_t) + 1 * sizeof(*p);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(57, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy3DPeerAsync_t* ms = TEE_CAST(ms_cudaMemcpy3DPeerAsync_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy3DPeerAsync_t), p, 1 * sizeof(*p));

	ms->ms_p = (struct cudaMemcpy3DPeerParms*)p;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy3DPeerAsync_t) + 1 * sizeof(*p);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(58, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemGetInfo_t* ms = TEE_CAST(ms_cudaMemGetInfo_t*, enclave_buffer);;
	

	ms->ms_free = free;
	ms->ms_total = total;
	bufsize = sizeof(ms_cudaMemGetInfo_t) + 1 * sizeof(*free) + 1 * sizeof(*total);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(59, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(free, enclave_buffer + sizeof(ms_cudaMemGetInfo_t), 1 * sizeof(*free));
		memcpy(total, enclave_buffer + sizeof(ms_cudaMemGetInfo_t) + 1 * sizeof(*free), 1 * sizeof(*total));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaArrayGetInfo_t* ms = TEE_CAST(ms_cudaArrayGetInfo_t*, enclave_buffer);;
	

	ms->ms_desc = desc;
	ms->ms_extent = extent;
	ms->ms_flags = flags;
	ms->ms_array = array;
	bufsize = sizeof(ms_cudaArrayGetInfo_t) + 1 * sizeof(*desc) + 1 * sizeof(*extent) + 1 * sizeof(*flags);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(60, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(desc, enclave_buffer + sizeof(ms_cudaArrayGetInfo_t), 1 * sizeof(*desc));
		memcpy(extent, enclave_buffer + sizeof(ms_cudaArrayGetInfo_t) + 1 * sizeof(*desc), 1 * sizeof(*extent));
		memcpy(flags, enclave_buffer + sizeof(ms_cudaArrayGetInfo_t) + 1 * sizeof(*desc) + 1 * sizeof(*extent), 1 * sizeof(*flags));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaArrayGetPlane_t* ms = TEE_CAST(ms_cudaArrayGetPlane_t*, enclave_buffer);;
	

	ms->ms_pPlaneArray = pPlaneArray;
	ms->ms_hArray = hArray;
	ms->ms_planeIdx = planeIdx;
	bufsize = sizeof(ms_cudaArrayGetPlane_t) + 1 * sizeof(*pPlaneArray);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(61, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(pPlaneArray, enclave_buffer + sizeof(ms_cudaArrayGetPlane_t), 1 * sizeof(*pPlaneArray));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaArray_t array)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaArrayGetSparseProperties_t* ms = TEE_CAST(ms_cudaArrayGetSparseProperties_t*, enclave_buffer);;
	

	ms->ms_sparseProperties = sparseProperties;
	ms->ms_array = array;
	bufsize = sizeof(ms_cudaArrayGetSparseProperties_t) + 1 * sizeof(*sparseProperties);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(62, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(sparseProperties, enclave_buffer + sizeof(ms_cudaArrayGetSparseProperties_t), 1 * sizeof(*sparseProperties));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMipmappedArrayGetSparseProperties_t* ms = TEE_CAST(ms_cudaMipmappedArrayGetSparseProperties_t*, enclave_buffer);;
	

	ms->ms_sparseProperties = sparseProperties;
	ms->ms_mipmap = mipmap;
	bufsize = sizeof(ms_cudaMipmappedArrayGetSparseProperties_t) + 1 * sizeof(*sparseProperties);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(63, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(sparseProperties, enclave_buffer + sizeof(ms_cudaMipmappedArrayGetSparseProperties_t), 1 * sizeof(*sparseProperties));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyNone(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyNone_t* ms = TEE_CAST(ms_cudaMemcpyNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyNone_t) + 0 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(64, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpySrc(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpySrc_t* ms = TEE_CAST(ms_cudaMemcpySrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpySrc_t) + 0, src, count);

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpySrc_t) + 0 + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(65, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyDst_t* ms = TEE_CAST(ms_cudaMemcpyDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyDst_t) + count + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(66, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpySrcDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpySrcDst_t* ms = TEE_CAST(ms_cudaMemcpySrcDst_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpySrcDst_t) + count, src, count);

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpySrcDst_t) + count + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(67, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpySrcDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyPeer_t* ms = TEE_CAST(ms_cudaMemcpyPeer_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dstDevice = dstDevice;
	ms->ms_src = (void*)src;
	ms->ms_srcDevice = srcDevice;
	ms->ms_count = count;
	bufsize = sizeof(ms_cudaMemcpyPeer_t) + 0 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(68, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DNone(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DNone_t* ms = TEE_CAST(ms_cudaMemcpy2DNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DNone_t) + 0 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(69, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DSrc(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DSrc_t* ms = TEE_CAST(ms_cudaMemcpy2DSrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy2DSrc_t) + 0, src, height * spitch);

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DSrc_t) + 0 + height * spitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(70, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DDst_t* ms = TEE_CAST(ms_cudaMemcpy2DDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DDst_t) + height * dpitch + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(71, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpy2DDst_t), height * dpitch);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DSrcDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DSrcDst_t* ms = TEE_CAST(ms_cudaMemcpy2DSrcDst_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy2DSrcDst_t) + height * dpitch, src, height * spitch);

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DSrcDst_t) + height * dpitch + height * spitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(72, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpy2DSrcDst_t), height * dpitch);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DToArrayNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DToArrayNone_t* ms = TEE_CAST(ms_cudaMemcpy2DToArrayNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DToArrayNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(73, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DToArraySrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DToArraySrc_t* ms = TEE_CAST(ms_cudaMemcpy2DToArraySrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy2DToArraySrc_t), src, height * spitch);

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DToArraySrc_t) + height * spitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(74, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DFromArrayNone(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DFromArrayNone_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DFromArrayNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(75, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DFromArrayDst(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DFromArrayDst_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DFromArrayDst_t) + height * dpitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(76, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpy2DFromArrayDst_t), height * dpitch);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DArrayToArray_t* ms = TEE_CAST(ms_cudaMemcpy2DArrayToArray_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_wOffsetDst = wOffsetDst;
	ms->ms_hOffsetDst = hOffsetDst;
	ms->ms_src = src;
	ms->ms_wOffsetSrc = wOffsetSrc;
	ms->ms_hOffsetSrc = hOffsetSrc;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpy2DArrayToArray_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(77, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToSymbolNone(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToSymbolNone_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolNone_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToSymbolNone_t), symbol, strlen((const char*)symbol) + 1);

	ms->ms_symbol = (void*)symbol;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyToSymbolNone_t) + strlen((const char*)symbol) + 1 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(78, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToSymbolSrc(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToSymbolSrc_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolSrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToSymbolSrc_t), symbol, strlen((const char*)symbol) + 1);
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToSymbolSrc_t) + strlen((const char*)symbol) + 1, src, count);

	ms->ms_symbol = (void*)symbol;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyToSymbolSrc_t) + strlen((const char*)symbol) + 1 + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(79, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromSymbolNone(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromSymbolNone_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolNone_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyFromSymbolNone_t) + 0, symbol, strlen((const char*)symbol) + 1);

	ms->ms_dst = dst;
	ms->ms_symbol = (void*)symbol;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyFromSymbolNone_t) + 0 + strlen((const char*)symbol) + 1;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(80, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromSymbolDst(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromSymbolDst_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolDst_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyFromSymbolDst_t) + count, symbol, strlen((const char*)symbol) + 1);

	ms->ms_dst = dst;
	ms->ms_symbol = (void*)symbol;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyFromSymbolDst_t) + count + strlen((const char*)symbol) + 1;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(81, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyFromSymbolDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyAsyncNone(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyAsyncNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyAsyncNone_t) + 0 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(82, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyAsyncSrc(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpyAsyncSrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyAsyncSrc_t) + 0, src, count);

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyAsyncSrc_t) + 0 + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(83, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyAsyncDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpyAsyncDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyAsyncDst_t) + count + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(84, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyAsyncDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyAsyncSrcDst(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyAsyncSrcDst_t* ms = TEE_CAST(ms_cudaMemcpyAsyncSrcDst_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyAsyncSrcDst_t) + count, src, count);

	ms->ms_dst = dst;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyAsyncSrcDst_t) + count + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(85, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyAsyncSrcDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyPeerAsync_t* ms = TEE_CAST(ms_cudaMemcpyPeerAsync_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dstDevice = dstDevice;
	ms->ms_src = (void*)src;
	ms->ms_srcDevice = srcDevice;
	ms->ms_count = count;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyPeerAsync_t) + 0 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(86, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DAsyncNone(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DAsyncNone_t) + 0 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(87, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DAsyncSrc(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncSrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy2DAsyncSrc_t) + 0, src, height * spitch);

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DAsyncSrc_t) + 0 + height * spitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(88, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DAsyncDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DAsyncDst_t) + height * dpitch + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(89, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpy2DAsyncDst_t), height * dpitch);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DAsyncSrcDst(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DAsyncSrcDst_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncSrcDst_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy2DAsyncSrcDst_t) + height * dpitch, src, height * spitch);

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DAsyncSrcDst_t) + height * dpitch + height * spitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(90, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpy2DAsyncSrcDst_t), height * dpitch);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DToArrayAsyncNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DToArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpy2DToArrayAsyncNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DToArrayAsyncNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(91, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DToArrayAsyncSrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DToArrayAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpy2DToArrayAsyncSrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpy2DToArrayAsyncSrc_t), src, height * spitch);

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_spitch = spitch;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DToArrayAsyncSrc_t) + height * spitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(92, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DFromArrayAsyncNone(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DFromArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayAsyncNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DFromArrayAsyncNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(93, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpy2DFromArrayAsyncDst(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpy2DFromArrayAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayAsyncDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_dpitch = dpitch;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpy2DFromArrayAsyncDst_t) + height * dpitch;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(94, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpy2DFromArrayAsyncDst_t), height * dpitch);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToSymbolAsyncNone(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToSymbolAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolAsyncNone_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToSymbolAsyncNone_t), symbol, strlen((const char*)symbol) + 1);

	ms->ms_symbol = (void*)symbol;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyToSymbolAsyncNone_t) + strlen((const char*)symbol) + 1 + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(95, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToSymbolAsyncSrc(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToSymbolAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolAsyncSrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToSymbolAsyncSrc_t), symbol, strlen((const char*)symbol) + 1);
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToSymbolAsyncSrc_t) + strlen((const char*)symbol) + 1, src, count);

	ms->ms_symbol = (void*)symbol;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyToSymbolAsyncSrc_t) + strlen((const char*)symbol) + 1 + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(96, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromSymbolAsyncNone(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromSymbolAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolAsyncNone_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyFromSymbolAsyncNone_t) + 0, symbol, strlen((const char*)symbol) + 1);

	ms->ms_dst = dst;
	ms->ms_symbol = (void*)symbol;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyFromSymbolAsyncNone_t) + 0 + strlen((const char*)symbol) + 1;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(97, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromSymbolAsyncDst(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromSymbolAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolAsyncDst_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyFromSymbolAsyncDst_t) + count, symbol, strlen((const char*)symbol) + 1);

	ms->ms_dst = dst;
	ms->ms_symbol = (void*)symbol;
	ms->ms_count = count;
	ms->ms_offset = offset;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyFromSymbolAsyncDst_t) + count + strlen((const char*)symbol) + 1;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(98, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyFromSymbolAsyncDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemset_t* ms = TEE_CAST(ms_cudaMemset_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_value = value;
	ms->ms_count = count;
	bufsize = sizeof(ms_cudaMemset_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(99, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemset2D_t* ms = TEE_CAST(ms_cudaMemset2D_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_pitch = pitch;
	ms->ms_value = value;
	ms->ms_width = width;
	ms->ms_height = height;
	bufsize = sizeof(ms_cudaMemset2D_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(100, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemset3D_t* ms = TEE_CAST(ms_cudaMemset3D_t*, enclave_buffer);;
	

	ms->ms_pitchedDevPtr = pitchedDevPtr;
	ms->ms_value = value;
	ms->ms_extent = extent;
	bufsize = sizeof(ms_cudaMemset3D_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(101, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemsetAsync_t* ms = TEE_CAST(ms_cudaMemsetAsync_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_value = value;
	ms->ms_count = count;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemsetAsync_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(102, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemset2DAsync_t* ms = TEE_CAST(ms_cudaMemset2DAsync_t*, enclave_buffer);;
	

	ms->ms_devPtr = devPtr;
	ms->ms_pitch = pitch;
	ms->ms_value = value;
	ms->ms_width = width;
	ms->ms_height = height;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemset2DAsync_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(103, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemset3DAsync_t* ms = TEE_CAST(ms_cudaMemset3DAsync_t*, enclave_buffer);;
	

	ms->ms_pitchedDevPtr = pitchedDevPtr;
	ms->ms_value = value;
	ms->ms_extent = extent;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemset3DAsync_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(104, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetSymbolAddress_t* ms = TEE_CAST(ms_cudaGetSymbolAddress_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaGetSymbolAddress_t) + 1 * sizeof(*devPtr), symbol, strlen((const char*)symbol) + 1);

	ms->ms_devPtr = devPtr;
	ms->ms_symbol = (void*)symbol;
	bufsize = sizeof(ms_cudaGetSymbolAddress_t) + 1 * sizeof(*devPtr) + strlen((const char*)symbol) + 1;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(105, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(devPtr, enclave_buffer + sizeof(ms_cudaGetSymbolAddress_t), 1 * sizeof(*devPtr));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaGetSymbolSize_t* ms = TEE_CAST(ms_cudaGetSymbolSize_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaGetSymbolSize_t) + 1 * sizeof(*size), symbol, strlen((const char*)symbol) + 1);

	ms->ms_size = size;
	ms->ms_symbol = (void*)symbol;
	bufsize = sizeof(ms_cudaGetSymbolSize_t) + 1 * sizeof(*size) + strlen((const char*)symbol) + 1;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(106, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(size, enclave_buffer + sizeof(ms_cudaGetSymbolSize_t), 1 * sizeof(*size));
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemPrefetchAsync_t* ms = TEE_CAST(ms_cudaMemPrefetchAsync_t*, enclave_buffer);;
	

	ms->ms_devPtr = (void*)devPtr;
	ms->ms_count = count;
	ms->ms_dstDevice = dstDevice;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemPrefetchAsync_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(107, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemAdvise_t* ms = TEE_CAST(ms_cudaMemAdvise_t*, enclave_buffer);;
	

	ms->ms_devPtr = (void*)devPtr;
	ms->ms_count = count;
	ms->ms_advice = advice;
	ms->ms_device = device;
	bufsize = sizeof(ms_cudaMemAdvise_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(108, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void* devPtr, size_t count)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemRangeGetAttribute_t* ms = TEE_CAST(ms_cudaMemRangeGetAttribute_t*, enclave_buffer);;
	

	ms->ms_data = data;
	ms->ms_dataSize = dataSize;
	ms->ms_attribute = attribute;
	ms->ms_devPtr = (void*)devPtr;
	ms->ms_count = count;
	bufsize = sizeof(ms_cudaMemRangeGetAttribute_t) + dataSize + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(109, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(data, enclave_buffer + sizeof(ms_cudaMemRangeGetAttribute_t), dataSize);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToArrayNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToArrayNone_t* ms = TEE_CAST(ms_cudaMemcpyToArrayNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyToArrayNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(110, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToArraySrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToArraySrc_t* ms = TEE_CAST(ms_cudaMemcpyToArraySrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToArraySrc_t), src, count);

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyToArraySrc_t) + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(111, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromArrayNone(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromArrayNone_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyFromArrayNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(112, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromArrayDst(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromArrayDst_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyFromArrayDst_t) + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(113, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyFromArrayDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyArrayToArray_t* ms = TEE_CAST(ms_cudaMemcpyArrayToArray_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_wOffsetDst = wOffsetDst;
	ms->ms_hOffsetDst = hOffsetDst;
	ms->ms_src = src;
	ms->ms_wOffsetSrc = wOffsetSrc;
	ms->ms_hOffsetSrc = hOffsetSrc;
	ms->ms_count = count;
	ms->ms_kind = kind;
	bufsize = sizeof(ms_cudaMemcpyArrayToArray_t);

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(114, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToArrayAsyncNone(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyToArrayAsyncNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyToArrayAsyncNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(115, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyToArrayAsyncSrc(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyToArrayAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpyToArrayAsyncSrc_t*, enclave_buffer);;
	
	memcpy(enclave_buffer + sizeof(ms_cudaMemcpyToArrayAsyncSrc_t), src, count);

	ms->ms_dst = dst;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_src = (void*)src;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyToArrayAsyncSrc_t) + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(116, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromArrayAsyncNone(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayAsyncNone_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyFromArrayAsyncNone_t) + 0;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(117, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

cudaError_t cudaMemcpyFromArrayAsyncDst(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TEE_Result status;
	int bufsize;
	char *enclave_buffer = rpc_buffer();
	cudaError_t retval;
	ms_cudaMemcpyFromArrayAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayAsyncDst_t*, enclave_buffer);;
	

	ms->ms_dst = dst;
	ms->ms_src = src;
	ms->ms_wOffset = wOffset;
	ms->ms_hOffset = hOffset;
	ms->ms_count = count;
	ms->ms_kind = kind;
	ms->ms_stream = stream;
	bufsize = sizeof(ms_cudaMemcpyFromArrayAsyncDst_t) + count;

	if(bufsize > buffer_size_in_bytes) {
		return cudaErrorMemoryValueTooLarge;
	}
	status = rpc_ecall(118, enclave_buffer, bufsize);

	if (status == TEE_SUCCESS) {
		retval = ms->ms_retval;
		memcpy(dst, enclave_buffer + sizeof(ms_cudaMemcpyFromArrayAsyncDst_t), count);
		RPC_DEBUG("ret -> %d (%d)", retval, status);
		return retval;
	}
	return cudaErrorInvalidValue;
}

