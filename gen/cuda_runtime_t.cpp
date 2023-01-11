#include "cuda_runtime_t.h"

#include "tee_internal_api.h"
#include "rpc/rpc.h"
#include <string.h> /* for memcpy etc */
#include <stdlib.h> /* for malloc/free etc */

typedef TEE_Result (*ecall_invoke_entry) (char* buffer);

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

static TEE_Result tee_cudaLaunchKernelByName(char *buffer)
{
	ms_cudaLaunchKernelByName_t* ms = TEE_CAST(ms_cudaLaunchKernelByName_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaLaunchKernelByName_t);

	TEE_Result status = TEE_SUCCESS;
	char* _tmp_funcname = TEE_CAST(char*, buffer_start + 0);
	size_t _len_funcname = _tmp_funcname ? strlen((const char*)(_tmp_funcname)) + 1 : 0;
	char* _in_funcname = NULL;
	void* _tmp_argbuf = TEE_CAST(void*, buffer_start + 0 + strlen((const char*)_tmp_funcname) + 1);
	int _tmp_argbufsize = ms->ms_argbufsize;
	size_t _len_argbuf = _tmp_argbufsize;
	void* _in_argbuf = NULL;
	uint32_t* _tmp_parameters = TEE_CAST(uint32_t*, buffer_start + 0 + strlen((const char*)_tmp_funcname) + 1 + _tmp_argbufsize);
	int _tmp_partotal_size = ms->ms_partotal_size;
	size_t _len_parameters = _tmp_partotal_size;
	uint32_t* _in_parameters = NULL;

	if (_tmp_funcname != NULL) {
		_in_funcname = (char*)malloc(_len_funcname);
		if (_in_funcname == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_funcname, _tmp_funcname, _len_funcname);
		((char*)_in_funcname)[_len_funcname - 1] = '\0';
	}
	if (_tmp_argbuf != NULL) {
		_in_argbuf = (void*)malloc(_len_argbuf);
		if (_in_argbuf == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_argbuf, _tmp_argbuf, _len_argbuf);
	}
	if (_tmp_parameters != NULL) {
		_in_parameters = (uint32_t*)malloc(_len_parameters);
		if (_in_parameters == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_parameters, _tmp_parameters, _len_parameters);
	}
	ms->ms_retval = cudaLaunchKernelByName(_in_funcname, ms->ms_gridDim, ms->ms_blockDim, _in_argbuf, _tmp_argbufsize, _in_parameters, _tmp_partotal_size, ms->ms_sharedMem, ms->ms_stream);
	RPC_SERVER_DEBUG("(%s, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_funcname, ms->ms_gridDim, ms->ms_blockDim, _in_argbuf, _tmp_argbufsize, _in_parameters, _tmp_partotal_size, ms->ms_sharedMem, ms->ms_stream, ms->ms_retval);
err:
	if (_in_funcname) free(_in_funcname);
	if (_in_argbuf) free(_in_argbuf);
	if (_in_parameters) free(_in_parameters);

	return status;
}

static TEE_Result tee_cudaThreadSynchronize(char *buffer)
{
	ms_cudaThreadSynchronize_t* ms = TEE_CAST(ms_cudaThreadSynchronize_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaThreadSynchronize_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaThreadSynchronize();
	RPC_SERVER_DEBUG("() => %lx" , ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaDeviceSynchronize(char *buffer)
{
	ms_cudaDeviceSynchronize_t* ms = TEE_CAST(ms_cudaDeviceSynchronize_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaDeviceSynchronize_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaDeviceSynchronize();
	RPC_SERVER_DEBUG("() => %lx" , ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaGetLastError(char *buffer)
{
	ms_cudaGetLastError_t* ms = TEE_CAST(ms_cudaGetLastError_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetLastError_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaGetLastError();
	RPC_SERVER_DEBUG("() => %lx" , ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaGetDeviceCount(char *buffer)
{
	ms_cudaGetDeviceCount_t* ms = TEE_CAST(ms_cudaGetDeviceCount_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetDeviceCount_t);

	TEE_Result status = TEE_SUCCESS;
	int* _tmp_count = TEE_CAST(int*, buffer_start + 0);
	size_t _len_count = 1 * sizeof(*_tmp_count);
	int* _in_count = NULL;

	if (_tmp_count != NULL) {
		if ((_in_count = (int*)malloc(_len_count)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_count, 0, _len_count);
	}
	ms->ms_retval = cudaGetDeviceCount(_in_count);
	RPC_SERVER_DEBUG("(%lx) => %lx", _in_count, ms->ms_retval);
err:
	if (_in_count) {
		memcpy(_tmp_count, _in_count, _len_count);
		free(_in_count);
	}

	return status;
}

static TEE_Result tee_cudaGetDeviceProperties(char *buffer)
{
	ms_cudaGetDeviceProperties_t* ms = TEE_CAST(ms_cudaGetDeviceProperties_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetDeviceProperties_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaDeviceProp* _tmp_prop = TEE_CAST(struct cudaDeviceProp*, buffer_start + 0);
	size_t _len_prop = 1 * sizeof(*_tmp_prop);
	struct cudaDeviceProp* _in_prop = NULL;

	if (_tmp_prop != NULL) {
		if ((_in_prop = (struct cudaDeviceProp*)malloc(_len_prop)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_prop, 0, _len_prop);
	}
	ms->ms_retval = cudaGetDeviceProperties(_in_prop, ms->ms_device);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _in_prop, ms->ms_device, ms->ms_retval);
err:
	if (_in_prop) {
		memcpy(_tmp_prop, _in_prop, _len_prop);
		free(_in_prop);
	}

	return status;
}

static TEE_Result tee_cudaDeviceGetAttribute(char *buffer)
{
	ms_cudaDeviceGetAttribute_t* ms = TEE_CAST(ms_cudaDeviceGetAttribute_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaDeviceGetAttribute_t);

	TEE_Result status = TEE_SUCCESS;
	int* _tmp_value = TEE_CAST(int*, buffer_start + 0);
	size_t _len_value = 1 * sizeof(*_tmp_value);
	int* _in_value = NULL;

	if (_tmp_value != NULL) {
		if ((_in_value = (int*)malloc(_len_value)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_value, 0, _len_value);
	}
	ms->ms_retval = cudaDeviceGetAttribute(_in_value, ms->ms_attr, ms->ms_device);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _in_value, ms->ms_attr, ms->ms_device, ms->ms_retval);
err:
	if (_in_value) {
		memcpy(_tmp_value, _in_value, _len_value);
		free(_in_value);
	}

	return status;
}

static TEE_Result tee_cudaChooseDevice(char *buffer)
{
	ms_cudaChooseDevice_t* ms = TEE_CAST(ms_cudaChooseDevice_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaChooseDevice_t);

	TEE_Result status = TEE_SUCCESS;
	int* _tmp_device = TEE_CAST(int*, buffer_start + 0);
	size_t _len_device = 1 * sizeof(*_tmp_device);
	int* _in_device = NULL;
	struct cudaDeviceProp* _tmp_prop = TEE_CAST(struct cudaDeviceProp*, buffer_start + 0 + 1 * sizeof(*_tmp_device));
	size_t _len_prop = 1 * sizeof(*_tmp_prop);
	struct cudaDeviceProp* _in_prop = NULL;

	if (_tmp_device != NULL) {
		if ((_in_device = (int*)malloc(_len_device)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_device, 0, _len_device);
	}
	if (_tmp_prop != NULL) {
		_in_prop = (struct cudaDeviceProp*)malloc(_len_prop);
		if (_in_prop == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_prop, _tmp_prop, _len_prop);
	}
	ms->ms_retval = cudaChooseDevice(_in_device, (const struct cudaDeviceProp*)_in_prop);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _in_device, (const struct cudaDeviceProp*)_in_prop, ms->ms_retval);
err:
	if (_in_device) {
		memcpy(_tmp_device, _in_device, _len_device);
		free(_in_device);
	}
	if (_in_prop) free((void*)_in_prop);

	return status;
}

static TEE_Result tee_cudaSetDevice(char *buffer)
{
	ms_cudaSetDevice_t* ms = TEE_CAST(ms_cudaSetDevice_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaSetDevice_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaSetDevice(ms->ms_device);
	RPC_SERVER_DEBUG("(%lx) => %lx", ms->ms_device, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaGetDevice(char *buffer)
{
	ms_cudaGetDevice_t* ms = TEE_CAST(ms_cudaGetDevice_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetDevice_t);

	TEE_Result status = TEE_SUCCESS;
	int* _tmp_device = TEE_CAST(int*, buffer_start + 0);
	size_t _len_device = 1 * sizeof(*_tmp_device);
	int* _in_device = NULL;

	if (_tmp_device != NULL) {
		if ((_in_device = (int*)malloc(_len_device)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_device, 0, _len_device);
	}
	ms->ms_retval = cudaGetDevice(_in_device);
	RPC_SERVER_DEBUG("(%lx) => %lx", _in_device, ms->ms_retval);
err:
	if (_in_device) {
		memcpy(_tmp_device, _in_device, _len_device);
		free(_in_device);
	}

	return status;
}

static TEE_Result tee_cudaSetValidDevices(char *buffer)
{
	ms_cudaSetValidDevices_t* ms = TEE_CAST(ms_cudaSetValidDevices_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaSetValidDevices_t);

	TEE_Result status = TEE_SUCCESS;
	int* _tmp_device_arr = TEE_CAST(int*, buffer_start + 0);
	int _tmp_len = ms->ms_len;
	size_t _len_device_arr = _tmp_len * sizeof(*_tmp_device_arr);
	int* _in_device_arr = NULL;

	if (_tmp_device_arr != NULL) {
		_in_device_arr = (int*)malloc(_len_device_arr);
		if (_in_device_arr == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_device_arr, _tmp_device_arr, _len_device_arr);
	}
	ms->ms_retval = cudaSetValidDevices(_in_device_arr, _tmp_len);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _in_device_arr, _tmp_len, ms->ms_retval);
err:
	if (_in_device_arr) free(_in_device_arr);

	return status;
}

static TEE_Result tee_cudaSetDeviceFlags(char *buffer)
{
	ms_cudaSetDeviceFlags_t* ms = TEE_CAST(ms_cudaSetDeviceFlags_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaSetDeviceFlags_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaSetDeviceFlags(ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx) => %lx", ms->ms_flags, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaGetDeviceFlags(char *buffer)
{
	ms_cudaGetDeviceFlags_t* ms = TEE_CAST(ms_cudaGetDeviceFlags_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetDeviceFlags_t);

	TEE_Result status = TEE_SUCCESS;
	unsigned int* _tmp_flags = TEE_CAST(unsigned int*, buffer_start + 0);
	size_t _len_flags = 1 * sizeof(*_tmp_flags);
	unsigned int* _in_flags = NULL;

	if (_tmp_flags != NULL) {
		if ((_in_flags = (unsigned int*)malloc(_len_flags)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_flags, 0, _len_flags);
	}
	ms->ms_retval = cudaGetDeviceFlags(_in_flags);
	RPC_SERVER_DEBUG("(%lx) => %lx", _in_flags, ms->ms_retval);
err:
	if (_in_flags) {
		memcpy(_tmp_flags, _in_flags, _len_flags);
		free(_in_flags);
	}

	return status;
}

static TEE_Result tee_cudaStreamCreate(char *buffer)
{
	ms_cudaStreamCreate_t* ms = TEE_CAST(ms_cudaStreamCreate_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamCreate_t);

	TEE_Result status = TEE_SUCCESS;
	cudaStream_t* _tmp_pStream = TEE_CAST(cudaStream_t*, buffer_start + 0);
	size_t _len_pStream = 1 * sizeof(*_tmp_pStream);
	cudaStream_t* _in_pStream = NULL;

	if (_tmp_pStream != NULL) {
		if ((_in_pStream = (cudaStream_t*)malloc(_len_pStream)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pStream, 0, _len_pStream);
	}
	ms->ms_retval = cudaStreamCreate(_in_pStream);
	RPC_SERVER_DEBUG("(%lx) => %lx", _in_pStream, ms->ms_retval);
err:
	if (_in_pStream) {
		memcpy(_tmp_pStream, _in_pStream, _len_pStream);
		free(_in_pStream);
	}

	return status;
}

static TEE_Result tee_cudaStreamCreateWithFlags(char *buffer)
{
	ms_cudaStreamCreateWithFlags_t* ms = TEE_CAST(ms_cudaStreamCreateWithFlags_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamCreateWithFlags_t);

	TEE_Result status = TEE_SUCCESS;
	cudaStream_t* _tmp_pStream = TEE_CAST(cudaStream_t*, buffer_start + 0);
	size_t _len_pStream = 1 * sizeof(*_tmp_pStream);
	cudaStream_t* _in_pStream = NULL;

	if (_tmp_pStream != NULL) {
		if ((_in_pStream = (cudaStream_t*)malloc(_len_pStream)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pStream, 0, _len_pStream);
	}
	ms->ms_retval = cudaStreamCreateWithFlags(_in_pStream, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _in_pStream, ms->ms_flags, ms->ms_retval);
err:
	if (_in_pStream) {
		memcpy(_tmp_pStream, _in_pStream, _len_pStream);
		free(_in_pStream);
	}

	return status;
}

static TEE_Result tee_cudaStreamCreateWithPriority(char *buffer)
{
	ms_cudaStreamCreateWithPriority_t* ms = TEE_CAST(ms_cudaStreamCreateWithPriority_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamCreateWithPriority_t);

	TEE_Result status = TEE_SUCCESS;
	cudaStream_t* _tmp_pStream = TEE_CAST(cudaStream_t*, buffer_start + 0);
	size_t _len_pStream = 1 * sizeof(*_tmp_pStream);
	cudaStream_t* _in_pStream = NULL;

	if (_tmp_pStream != NULL) {
		if ((_in_pStream = (cudaStream_t*)malloc(_len_pStream)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pStream, 0, _len_pStream);
	}
	ms->ms_retval = cudaStreamCreateWithPriority(_in_pStream, ms->ms_flags, ms->ms_priority);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _in_pStream, ms->ms_flags, ms->ms_priority, ms->ms_retval);
err:
	if (_in_pStream) {
		memcpy(_tmp_pStream, _in_pStream, _len_pStream);
		free(_in_pStream);
	}

	return status;
}

static TEE_Result tee_cudaStreamGetPriority(char *buffer)
{
	ms_cudaStreamGetPriority_t* ms = TEE_CAST(ms_cudaStreamGetPriority_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamGetPriority_t);

	TEE_Result status = TEE_SUCCESS;
	int* _tmp_priority = TEE_CAST(int*, buffer_start + 0);
	size_t _len_priority = 1 * sizeof(*_tmp_priority);
	int* _in_priority = NULL;

	if (_tmp_priority != NULL) {
		if ((_in_priority = (int*)malloc(_len_priority)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_priority, 0, _len_priority);
	}
	ms->ms_retval = cudaStreamGetPriority(ms->ms_hStream, _in_priority);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", ms->ms_hStream, _in_priority, ms->ms_retval);
err:
	if (_in_priority) {
		memcpy(_tmp_priority, _in_priority, _len_priority);
		free(_in_priority);
	}

	return status;
}

static TEE_Result tee_cudaStreamGetFlags(char *buffer)
{
	ms_cudaStreamGetFlags_t* ms = TEE_CAST(ms_cudaStreamGetFlags_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamGetFlags_t);

	TEE_Result status = TEE_SUCCESS;
	unsigned int* _tmp_flags = TEE_CAST(unsigned int*, buffer_start + 0);
	size_t _len_flags = 1 * sizeof(*_tmp_flags);
	unsigned int* _in_flags = NULL;

	if (_tmp_flags != NULL) {
		if ((_in_flags = (unsigned int*)malloc(_len_flags)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_flags, 0, _len_flags);
	}
	ms->ms_retval = cudaStreamGetFlags(ms->ms_hStream, _in_flags);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", ms->ms_hStream, _in_flags, ms->ms_retval);
err:
	if (_in_flags) {
		memcpy(_tmp_flags, _in_flags, _len_flags);
		free(_in_flags);
	}

	return status;
}

static TEE_Result tee_cudaStreamDestroy(char *buffer)
{
	ms_cudaStreamDestroy_t* ms = TEE_CAST(ms_cudaStreamDestroy_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamDestroy_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaStreamDestroy(ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx) => %lx", ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaStreamWaitEvent(char *buffer)
{
	ms_cudaStreamWaitEvent_t* ms = TEE_CAST(ms_cudaStreamWaitEvent_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamWaitEvent_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaStreamWaitEvent(ms->ms_stream, ms->ms_event, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", ms->ms_stream, ms->ms_event, ms->ms_flags, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaStreamSynchronize(char *buffer)
{
	ms_cudaStreamSynchronize_t* ms = TEE_CAST(ms_cudaStreamSynchronize_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamSynchronize_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaStreamSynchronize(ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx) => %lx", ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaStreamQuery(char *buffer)
{
	ms_cudaStreamQuery_t* ms = TEE_CAST(ms_cudaStreamQuery_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamQuery_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaStreamQuery(ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx) => %lx", ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaStreamAttachMemAsync(char *buffer)
{
	ms_cudaStreamAttachMemAsync_t* ms = TEE_CAST(ms_cudaStreamAttachMemAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamAttachMemAsync_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaStreamAttachMemAsync(ms->ms_stream, _tmp_devPtr, ms->ms_length, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", ms->ms_stream, _tmp_devPtr, ms->ms_length, ms->ms_flags, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaStreamBeginCapture(char *buffer)
{
	ms_cudaStreamBeginCapture_t* ms = TEE_CAST(ms_cudaStreamBeginCapture_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamBeginCapture_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaStreamBeginCapture(ms->ms_stream, ms->ms_mode);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", ms->ms_stream, ms->ms_mode, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaThreadExchangeStreamCaptureMode(char *buffer)
{
	ms_cudaThreadExchangeStreamCaptureMode_t* ms = TEE_CAST(ms_cudaThreadExchangeStreamCaptureMode_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaThreadExchangeStreamCaptureMode_t);

	TEE_Result status = TEE_SUCCESS;
	enum cudaStreamCaptureMode* _tmp_mode = TEE_CAST(enum cudaStreamCaptureMode*, buffer_start + 0);
	size_t _len_mode = 1 * sizeof(*_tmp_mode);
	enum cudaStreamCaptureMode* _in_mode = NULL;

	if (_tmp_mode != NULL) {
		_in_mode = (enum cudaStreamCaptureMode*)malloc(_len_mode);
		if (_in_mode == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy(_in_mode, _tmp_mode, _len_mode);
	}
	ms->ms_retval = cudaThreadExchangeStreamCaptureMode(_in_mode);
	RPC_SERVER_DEBUG("(%lx) => %lx", _in_mode, ms->ms_retval);
err:
	if (_in_mode) {
		memcpy(_tmp_mode, _in_mode, _len_mode);
		free(_in_mode);
	}

	return status;
}

static TEE_Result tee_cudaStreamEndCapture(char *buffer)
{
	ms_cudaStreamEndCapture_t* ms = TEE_CAST(ms_cudaStreamEndCapture_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamEndCapture_t);

	TEE_Result status = TEE_SUCCESS;
	cudaGraph_t* _tmp_pGraph = TEE_CAST(cudaGraph_t*, buffer_start + 0);
	size_t _len_pGraph = 1 * sizeof(*_tmp_pGraph);
	cudaGraph_t* _in_pGraph = NULL;

	if (_tmp_pGraph != NULL) {
		if ((_in_pGraph = (cudaGraph_t*)malloc(_len_pGraph)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pGraph, 0, _len_pGraph);
	}
	ms->ms_retval = cudaStreamEndCapture(ms->ms_stream, _in_pGraph);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", ms->ms_stream, _in_pGraph, ms->ms_retval);
err:
	if (_in_pGraph) {
		memcpy(_tmp_pGraph, _in_pGraph, _len_pGraph);
		free(_in_pGraph);
	}

	return status;
}

static TEE_Result tee_cudaStreamIsCapturing(char *buffer)
{
	ms_cudaStreamIsCapturing_t* ms = TEE_CAST(ms_cudaStreamIsCapturing_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamIsCapturing_t);

	TEE_Result status = TEE_SUCCESS;
	enum cudaStreamCaptureStatus* _tmp_pCaptureStatus = TEE_CAST(enum cudaStreamCaptureStatus*, buffer_start + 0);
	size_t _len_pCaptureStatus = 1 * sizeof(*_tmp_pCaptureStatus);
	enum cudaStreamCaptureStatus* _in_pCaptureStatus = NULL;

	if (_tmp_pCaptureStatus != NULL) {
		if ((_in_pCaptureStatus = (enum cudaStreamCaptureStatus*)malloc(_len_pCaptureStatus)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pCaptureStatus, 0, _len_pCaptureStatus);
	}
	ms->ms_retval = cudaStreamIsCapturing(ms->ms_stream, _in_pCaptureStatus);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", ms->ms_stream, _in_pCaptureStatus, ms->ms_retval);
err:
	if (_in_pCaptureStatus) {
		memcpy(_tmp_pCaptureStatus, _in_pCaptureStatus, _len_pCaptureStatus);
		free(_in_pCaptureStatus);
	}

	return status;
}

static TEE_Result tee_cudaStreamGetCaptureInfo(char *buffer)
{
	ms_cudaStreamGetCaptureInfo_t* ms = TEE_CAST(ms_cudaStreamGetCaptureInfo_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaStreamGetCaptureInfo_t);

	TEE_Result status = TEE_SUCCESS;
	enum cudaStreamCaptureStatus* _tmp_pCaptureStatus = TEE_CAST(enum cudaStreamCaptureStatus*, buffer_start + 0);
	size_t _len_pCaptureStatus = 1 * sizeof(*_tmp_pCaptureStatus);
	enum cudaStreamCaptureStatus* _in_pCaptureStatus = NULL;
	unsigned long long* _tmp_pId = TEE_CAST(unsigned long long*, buffer_start + 0 + 1 * sizeof(*_tmp_pCaptureStatus));
	size_t _len_pId = 1 * sizeof(*_tmp_pId);
	unsigned long long* _in_pId = NULL;

	if (_tmp_pCaptureStatus != NULL) {
		if ((_in_pCaptureStatus = (enum cudaStreamCaptureStatus*)malloc(_len_pCaptureStatus)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pCaptureStatus, 0, _len_pCaptureStatus);
	}
	if (_tmp_pId != NULL) {
		if ((_in_pId = (unsigned long long*)malloc(_len_pId)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pId, 0, _len_pId);
	}
	ms->ms_retval = cudaStreamGetCaptureInfo(ms->ms_stream, _in_pCaptureStatus, _in_pId);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", ms->ms_stream, _in_pCaptureStatus, _in_pId, ms->ms_retval);
err:
	if (_in_pCaptureStatus) {
		memcpy(_tmp_pCaptureStatus, _in_pCaptureStatus, _len_pCaptureStatus);
		free(_in_pCaptureStatus);
	}
	if (_in_pId) {
		memcpy(_tmp_pId, _in_pId, _len_pId);
		free(_in_pId);
	}

	return status;
}

static TEE_Result tee_cudaMallocManaged(char *buffer)
{
	ms_cudaMallocManaged_t* ms = TEE_CAST(ms_cudaMallocManaged_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMallocManaged_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_devPtr = TEE_CAST(void**, buffer_start + 0);
	size_t _len_devPtr = 1 * sizeof(*_tmp_devPtr);
	void** _in_devPtr = NULL;

	if (_tmp_devPtr != NULL) {
		if ((_in_devPtr = (void**)malloc(_len_devPtr)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_devPtr, 0, _len_devPtr);
	}
	ms->ms_retval = cudaMallocManaged(_in_devPtr, ms->ms_size, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _in_devPtr, ms->ms_size, ms->ms_flags, ms->ms_retval);
err:
	if (_in_devPtr) {
		memcpy(_tmp_devPtr, _in_devPtr, _len_devPtr);
		free(_in_devPtr);
	}

	return status;
}

static TEE_Result tee_cudaMalloc(char *buffer)
{
	ms_cudaMalloc_t* ms = TEE_CAST(ms_cudaMalloc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMalloc_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaMalloc(_tmp_devPtr, ms->ms_size);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _tmp_devPtr, ms->ms_size, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMallocHost(char *buffer)
{
	ms_cudaMallocHost_t* ms = TEE_CAST(ms_cudaMallocHost_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMallocHost_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_ptr = ms->ms_ptr;

	ms->ms_retval = cudaMallocHost(_tmp_ptr, ms->ms_size);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _tmp_ptr, ms->ms_size, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMallocPitch(char *buffer)
{
	ms_cudaMallocPitch_t* ms = TEE_CAST(ms_cudaMallocPitch_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMallocPitch_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_devPtr = ms->ms_devPtr;
	size_t* _tmp_pitch = TEE_CAST(size_t*, buffer_start + 0 + 0);
	size_t _len_pitch = 1 * sizeof(*_tmp_pitch);
	size_t* _in_pitch = NULL;

	if (_tmp_pitch != NULL) {
		if ((_in_pitch = (size_t*)malloc(_len_pitch)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pitch, 0, _len_pitch);
	}
	ms->ms_retval = cudaMallocPitch(_tmp_devPtr, _in_pitch, ms->ms_width, ms->ms_height);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _tmp_devPtr, _in_pitch, ms->ms_width, ms->ms_height, ms->ms_retval);
err:
	if (_in_pitch) {
		memcpy(_tmp_pitch, _in_pitch, _len_pitch);
		free(_in_pitch);
	}

	return status;
}

static TEE_Result tee_cudaMallocArray(char *buffer)
{
	ms_cudaMallocArray_t* ms = TEE_CAST(ms_cudaMallocArray_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMallocArray_t);

	TEE_Result status = TEE_SUCCESS;
	cudaArray_t* _tmp_array = ms->ms_array;
	struct cudaChannelFormatDesc* _tmp_desc = TEE_CAST(struct cudaChannelFormatDesc*, buffer_start + 0 + 0);
	size_t _len_desc = 1 * sizeof(*_tmp_desc);
	struct cudaChannelFormatDesc* _in_desc = NULL;

	if (_tmp_desc != NULL) {
		_in_desc = (struct cudaChannelFormatDesc*)malloc(_len_desc);
		if (_in_desc == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_desc, _tmp_desc, _len_desc);
	}
	ms->ms_retval = cudaMallocArray(_tmp_array, (const struct cudaChannelFormatDesc*)_in_desc, ms->ms_width, ms->ms_height, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _tmp_array, (const struct cudaChannelFormatDesc*)_in_desc, ms->ms_width, ms->ms_height, ms->ms_flags, ms->ms_retval);
err:
	if (_in_desc) free((void*)_in_desc);

	return status;
}

static TEE_Result tee_cudaFree(char *buffer)
{
	ms_cudaFree_t* ms = TEE_CAST(ms_cudaFree_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaFree_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaFree(_tmp_devPtr);
	RPC_SERVER_DEBUG("(%lx) => %lx", _tmp_devPtr, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaFreeHost(char *buffer)
{
	ms_cudaFreeHost_t* ms = TEE_CAST(ms_cudaFreeHost_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaFreeHost_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_ptr = ms->ms_ptr;

	ms->ms_retval = cudaFreeHost(_tmp_ptr);
	RPC_SERVER_DEBUG("(%lx) => %lx", _tmp_ptr, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaFreeArray(char *buffer)
{
	ms_cudaFreeArray_t* ms = TEE_CAST(ms_cudaFreeArray_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaFreeArray_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaFreeArray(ms->ms_array);
	RPC_SERVER_DEBUG("(%lx) => %lx", ms->ms_array, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaFreeMipmappedArray(char *buffer)
{
	ms_cudaFreeMipmappedArray_t* ms = TEE_CAST(ms_cudaFreeMipmappedArray_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaFreeMipmappedArray_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaFreeMipmappedArray(ms->ms_mipmappedArray);
	RPC_SERVER_DEBUG("(%lx) => %lx", ms->ms_mipmappedArray, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaHostAlloc(char *buffer)
{
	ms_cudaHostAlloc_t* ms = TEE_CAST(ms_cudaHostAlloc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaHostAlloc_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_pHost = TEE_CAST(void**, buffer_start + 0);
	size_t _len_pHost = 1 * sizeof(*_tmp_pHost);
	void** _in_pHost = NULL;

	if (_tmp_pHost != NULL) {
		if ((_in_pHost = (void**)malloc(_len_pHost)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pHost, 0, _len_pHost);
	}
	ms->ms_retval = cudaHostAlloc(_in_pHost, ms->ms_size, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _in_pHost, ms->ms_size, ms->ms_flags, ms->ms_retval);
err:
	if (_in_pHost) {
		memcpy(_tmp_pHost, _in_pHost, _len_pHost);
		free(_in_pHost);
	}

	return status;
}

static TEE_Result tee_cudaHostRegister(char *buffer)
{
	ms_cudaHostRegister_t* ms = TEE_CAST(ms_cudaHostRegister_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaHostRegister_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_ptr = ms->ms_ptr;

	ms->ms_retval = cudaHostRegister(_tmp_ptr, ms->ms_size, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _tmp_ptr, ms->ms_size, ms->ms_flags, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaHostUnregister(char *buffer)
{
	ms_cudaHostUnregister_t* ms = TEE_CAST(ms_cudaHostUnregister_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaHostUnregister_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_ptr = ms->ms_ptr;

	ms->ms_retval = cudaHostUnregister(_tmp_ptr);
	RPC_SERVER_DEBUG("(%lx) => %lx", _tmp_ptr, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaHostGetDevicePointer(char *buffer)
{
	ms_cudaHostGetDevicePointer_t* ms = TEE_CAST(ms_cudaHostGetDevicePointer_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaHostGetDevicePointer_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_pDevice = TEE_CAST(void**, buffer_start + 0);
	size_t _len_pDevice = 1 * sizeof(*_tmp_pDevice);
	void** _in_pDevice = NULL;
	void* _tmp_pHost = ms->ms_pHost;

	if (_tmp_pDevice != NULL) {
		if ((_in_pDevice = (void**)malloc(_len_pDevice)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pDevice, 0, _len_pDevice);
	}
	ms->ms_retval = cudaHostGetDevicePointer(_in_pDevice, _tmp_pHost, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _in_pDevice, _tmp_pHost, ms->ms_flags, ms->ms_retval);
err:
	if (_in_pDevice) {
		memcpy(_tmp_pDevice, _in_pDevice, _len_pDevice);
		free(_in_pDevice);
	}

	return status;
}

static TEE_Result tee_cudaHostGetFlags(char *buffer)
{
	ms_cudaHostGetFlags_t* ms = TEE_CAST(ms_cudaHostGetFlags_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaHostGetFlags_t);

	TEE_Result status = TEE_SUCCESS;
	unsigned int* _tmp_pFlags = TEE_CAST(unsigned int*, buffer_start + 0);
	size_t _len_pFlags = 1 * sizeof(*_tmp_pFlags);
	unsigned int* _in_pFlags = NULL;
	void* _tmp_pHost = ms->ms_pHost;

	if (_tmp_pFlags != NULL) {
		if ((_in_pFlags = (unsigned int*)malloc(_len_pFlags)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pFlags, 0, _len_pFlags);
	}
	ms->ms_retval = cudaHostGetFlags(_in_pFlags, _tmp_pHost);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _in_pFlags, _tmp_pHost, ms->ms_retval);
err:
	if (_in_pFlags) {
		memcpy(_tmp_pFlags, _in_pFlags, _len_pFlags);
		free(_in_pFlags);
	}

	return status;
}

static TEE_Result tee_cudaMalloc3D(char *buffer)
{
	ms_cudaMalloc3D_t* ms = TEE_CAST(ms_cudaMalloc3D_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMalloc3D_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaPitchedPtr* _tmp_pitchedDevPtr = TEE_CAST(struct cudaPitchedPtr*, buffer_start + 0);
	size_t _len_pitchedDevPtr = 1 * sizeof(*_tmp_pitchedDevPtr);
	struct cudaPitchedPtr* _in_pitchedDevPtr = NULL;

	if (_tmp_pitchedDevPtr != NULL) {
		if ((_in_pitchedDevPtr = (struct cudaPitchedPtr*)malloc(_len_pitchedDevPtr)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_pitchedDevPtr, 0, _len_pitchedDevPtr);
	}
	ms->ms_retval = cudaMalloc3D(_in_pitchedDevPtr, ms->ms_extent);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _in_pitchedDevPtr, ms->ms_extent, ms->ms_retval);
err:
	if (_in_pitchedDevPtr) {
		memcpy(_tmp_pitchedDevPtr, _in_pitchedDevPtr, _len_pitchedDevPtr);
		free(_in_pitchedDevPtr);
	}

	return status;
}

static TEE_Result tee_cudaMalloc3DArray(char *buffer)
{
	ms_cudaMalloc3DArray_t* ms = TEE_CAST(ms_cudaMalloc3DArray_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMalloc3DArray_t);

	TEE_Result status = TEE_SUCCESS;
	cudaArray_t* _tmp_array = TEE_CAST(cudaArray_t*, buffer_start + 0);
	size_t _len_array = 1 * sizeof(*_tmp_array);
	cudaArray_t* _in_array = NULL;
	struct cudaChannelFormatDesc* _tmp_desc = TEE_CAST(struct cudaChannelFormatDesc*, buffer_start + 0 + 1 * sizeof(*_tmp_array));
	size_t _len_desc = 1 * sizeof(*_tmp_desc);
	struct cudaChannelFormatDesc* _in_desc = NULL;

	if (_tmp_array != NULL) {
		if ((_in_array = (cudaArray_t*)malloc(_len_array)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_array, 0, _len_array);
	}
	if (_tmp_desc != NULL) {
		_in_desc = (struct cudaChannelFormatDesc*)malloc(_len_desc);
		if (_in_desc == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_desc, _tmp_desc, _len_desc);
	}
	ms->ms_retval = cudaMalloc3DArray(_in_array, (const struct cudaChannelFormatDesc*)_in_desc, ms->ms_extent, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _in_array, (const struct cudaChannelFormatDesc*)_in_desc, ms->ms_extent, ms->ms_flags, ms->ms_retval);
err:
	if (_in_array) {
		memcpy(_tmp_array, _in_array, _len_array);
		free(_in_array);
	}
	if (_in_desc) free((void*)_in_desc);

	return status;
}

static TEE_Result tee_cudaMallocMipmappedArray(char *buffer)
{
	ms_cudaMallocMipmappedArray_t* ms = TEE_CAST(ms_cudaMallocMipmappedArray_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMallocMipmappedArray_t);

	TEE_Result status = TEE_SUCCESS;
	cudaMipmappedArray_t* _tmp_mipmappedArray = TEE_CAST(cudaMipmappedArray_t*, buffer_start + 0);
	size_t _len_mipmappedArray = 1 * sizeof(*_tmp_mipmappedArray);
	cudaMipmappedArray_t* _in_mipmappedArray = NULL;
	struct cudaChannelFormatDesc* _tmp_desc = TEE_CAST(struct cudaChannelFormatDesc*, buffer_start + 0 + 1 * sizeof(*_tmp_mipmappedArray));
	size_t _len_desc = 1 * sizeof(*_tmp_desc);
	struct cudaChannelFormatDesc* _in_desc = NULL;

	if (_tmp_mipmappedArray != NULL) {
		if ((_in_mipmappedArray = (cudaMipmappedArray_t*)malloc(_len_mipmappedArray)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_mipmappedArray, 0, _len_mipmappedArray);
	}
	if (_tmp_desc != NULL) {
		_in_desc = (struct cudaChannelFormatDesc*)malloc(_len_desc);
		if (_in_desc == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_desc, _tmp_desc, _len_desc);
	}
	ms->ms_retval = cudaMallocMipmappedArray(_in_mipmappedArray, (const struct cudaChannelFormatDesc*)_in_desc, ms->ms_extent, ms->ms_numLevels, ms->ms_flags);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _in_mipmappedArray, (const struct cudaChannelFormatDesc*)_in_desc, ms->ms_extent, ms->ms_numLevels, ms->ms_flags, ms->ms_retval);
err:
	if (_in_mipmappedArray) {
		memcpy(_tmp_mipmappedArray, _in_mipmappedArray, _len_mipmappedArray);
		free(_in_mipmappedArray);
	}
	if (_in_desc) free((void*)_in_desc);

	return status;
}

static TEE_Result tee_cudaGetMipmappedArrayLevel(char *buffer)
{
	ms_cudaGetMipmappedArrayLevel_t* ms = TEE_CAST(ms_cudaGetMipmappedArrayLevel_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetMipmappedArrayLevel_t);

	TEE_Result status = TEE_SUCCESS;
	cudaArray_t* _tmp_levelArray = TEE_CAST(cudaArray_t*, buffer_start + 0);
	size_t _len_levelArray = 1 * sizeof(*_tmp_levelArray);
	cudaArray_t* _in_levelArray = NULL;

	if (_tmp_levelArray != NULL) {
		if ((_in_levelArray = (cudaArray_t*)malloc(_len_levelArray)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_levelArray, 0, _len_levelArray);
	}
	ms->ms_retval = cudaGetMipmappedArrayLevel(_in_levelArray, ms->ms_mipmappedArray, ms->ms_level);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _in_levelArray, ms->ms_mipmappedArray, ms->ms_level, ms->ms_retval);
err:
	if (_in_levelArray) {
		memcpy(_tmp_levelArray, _in_levelArray, _len_levelArray);
		free(_in_levelArray);
	}

	return status;
}

static TEE_Result tee_cudaMemcpy3D(char *buffer)
{
	ms_cudaMemcpy3D_t* ms = TEE_CAST(ms_cudaMemcpy3D_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy3D_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaMemcpy3DParms* _tmp_p = TEE_CAST(struct cudaMemcpy3DParms*, buffer_start + 0);
	size_t _len_p = 1 * sizeof(*_tmp_p);
	struct cudaMemcpy3DParms* _in_p = NULL;

	if (_tmp_p != NULL) {
		_in_p = (struct cudaMemcpy3DParms*)malloc(_len_p);
		if (_in_p == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_p, _tmp_p, _len_p);
	}
	ms->ms_retval = cudaMemcpy3D((const struct cudaMemcpy3DParms*)_in_p);
	RPC_SERVER_DEBUG("(%lx) => %lx", (const struct cudaMemcpy3DParms*)_in_p, ms->ms_retval);
err:
	if (_in_p) free((void*)_in_p);

	return status;
}

static TEE_Result tee_cudaMemcpy3DPeer(char *buffer)
{
	ms_cudaMemcpy3DPeer_t* ms = TEE_CAST(ms_cudaMemcpy3DPeer_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy3DPeer_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaMemcpy3DPeerParms* _tmp_p = TEE_CAST(struct cudaMemcpy3DPeerParms*, buffer_start + 0);
	size_t _len_p = 1 * sizeof(*_tmp_p);
	struct cudaMemcpy3DPeerParms* _in_p = NULL;

	if (_tmp_p != NULL) {
		_in_p = (struct cudaMemcpy3DPeerParms*)malloc(_len_p);
		if (_in_p == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_p, _tmp_p, _len_p);
	}
	ms->ms_retval = cudaMemcpy3DPeer((const struct cudaMemcpy3DPeerParms*)_in_p);
	RPC_SERVER_DEBUG("(%lx) => %lx", (const struct cudaMemcpy3DPeerParms*)_in_p, ms->ms_retval);
err:
	if (_in_p) free((void*)_in_p);

	return status;
}

static TEE_Result tee_cudaMemcpy3DAsync(char *buffer)
{
	ms_cudaMemcpy3DAsync_t* ms = TEE_CAST(ms_cudaMemcpy3DAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy3DAsync_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaMemcpy3DParms* _tmp_p = TEE_CAST(struct cudaMemcpy3DParms*, buffer_start + 0);
	size_t _len_p = 1 * sizeof(*_tmp_p);
	struct cudaMemcpy3DParms* _in_p = NULL;

	if (_tmp_p != NULL) {
		_in_p = (struct cudaMemcpy3DParms*)malloc(_len_p);
		if (_in_p == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_p, _tmp_p, _len_p);
	}
	ms->ms_retval = cudaMemcpy3DAsync((const struct cudaMemcpy3DParms*)_in_p, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", (const struct cudaMemcpy3DParms*)_in_p, ms->ms_stream, ms->ms_retval);
err:
	if (_in_p) free((void*)_in_p);

	return status;
}

static TEE_Result tee_cudaMemcpy3DPeerAsync(char *buffer)
{
	ms_cudaMemcpy3DPeerAsync_t* ms = TEE_CAST(ms_cudaMemcpy3DPeerAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy3DPeerAsync_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaMemcpy3DPeerParms* _tmp_p = TEE_CAST(struct cudaMemcpy3DPeerParms*, buffer_start + 0);
	size_t _len_p = 1 * sizeof(*_tmp_p);
	struct cudaMemcpy3DPeerParms* _in_p = NULL;

	if (_tmp_p != NULL) {
		_in_p = (struct cudaMemcpy3DPeerParms*)malloc(_len_p);
		if (_in_p == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_p, _tmp_p, _len_p);
	}
	ms->ms_retval = cudaMemcpy3DPeerAsync((const struct cudaMemcpy3DPeerParms*)_in_p, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", (const struct cudaMemcpy3DPeerParms*)_in_p, ms->ms_stream, ms->ms_retval);
err:
	if (_in_p) free((void*)_in_p);

	return status;
}

static TEE_Result tee_cudaMemGetInfo(char *buffer)
{
	ms_cudaMemGetInfo_t* ms = TEE_CAST(ms_cudaMemGetInfo_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemGetInfo_t);

	TEE_Result status = TEE_SUCCESS;
	size_t* _tmp_free = TEE_CAST(size_t*, buffer_start + 0);
	size_t _len_free = 1 * sizeof(*_tmp_free);
	size_t* _in_free = NULL;
	size_t* _tmp_total = TEE_CAST(size_t*, buffer_start + 0 + 1 * sizeof(*_tmp_free));
	size_t _len_total = 1 * sizeof(*_tmp_total);
	size_t* _in_total = NULL;

	if (_tmp_free != NULL) {
		if ((_in_free = (size_t*)malloc(_len_free)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_free, 0, _len_free);
	}
	if (_tmp_total != NULL) {
		if ((_in_total = (size_t*)malloc(_len_total)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_total, 0, _len_total);
	}
	ms->ms_retval = cudaMemGetInfo(_in_free, _in_total);
	RPC_SERVER_DEBUG("(%lx, %lx) => %lx", _in_free, _in_total, ms->ms_retval);
err:
	if (_in_free) {
		memcpy(_tmp_free, _in_free, _len_free);
		free(_in_free);
	}
	if (_in_total) {
		memcpy(_tmp_total, _in_total, _len_total);
		free(_in_total);
	}

	return status;
}

static TEE_Result tee_cudaArrayGetInfo(char *buffer)
{
	ms_cudaArrayGetInfo_t* ms = TEE_CAST(ms_cudaArrayGetInfo_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaArrayGetInfo_t);

	TEE_Result status = TEE_SUCCESS;
	struct cudaChannelFormatDesc* _tmp_desc = TEE_CAST(struct cudaChannelFormatDesc*, buffer_start + 0);
	size_t _len_desc = 1 * sizeof(*_tmp_desc);
	struct cudaChannelFormatDesc* _in_desc = NULL;
	struct cudaExtent* _tmp_extent = TEE_CAST(struct cudaExtent*, buffer_start + 0 + 1 * sizeof(*_tmp_desc));
	size_t _len_extent = 1 * sizeof(*_tmp_extent);
	struct cudaExtent* _in_extent = NULL;
	unsigned int* _tmp_flags = TEE_CAST(unsigned int*, buffer_start + 0 + 1 * sizeof(*_tmp_desc) + 1 * sizeof(*_tmp_extent));
	size_t _len_flags = 1 * sizeof(*_tmp_flags);
	unsigned int* _in_flags = NULL;

	if (_tmp_desc != NULL) {
		if ((_in_desc = (struct cudaChannelFormatDesc*)malloc(_len_desc)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_desc, 0, _len_desc);
	}
	if (_tmp_extent != NULL) {
		if ((_in_extent = (struct cudaExtent*)malloc(_len_extent)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_extent, 0, _len_extent);
	}
	if (_tmp_flags != NULL) {
		if ((_in_flags = (unsigned int*)malloc(_len_flags)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_flags, 0, _len_flags);
	}
	ms->ms_retval = cudaArrayGetInfo(_in_desc, _in_extent, _in_flags, ms->ms_array);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _in_desc, _in_extent, _in_flags, ms->ms_array, ms->ms_retval);
err:
	if (_in_desc) {
		memcpy(_tmp_desc, _in_desc, _len_desc);
		free(_in_desc);
	}
	if (_in_extent) {
		memcpy(_tmp_extent, _in_extent, _len_extent);
		free(_in_extent);
	}
	if (_in_flags) {
		memcpy(_tmp_flags, _in_flags, _len_flags);
		free(_in_flags);
	}

	return status;
}

static TEE_Result tee_cudaMemcpyNone(char *buffer)
{
	ms_cudaMemcpyNone_t* ms = TEE_CAST(ms_cudaMemcpyNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpyNone(_tmp_dst, (const void*)_tmp_src, ms->ms_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _tmp_dst, (const void*)_tmp_src, ms->ms_count, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpySrc(char *buffer)
{
	ms_cudaMemcpySrc_t* ms = TEE_CAST(ms_cudaMemcpySrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpySrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpySrc(_tmp_dst, (const void*)_in_src, _tmp_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _tmp_dst, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyDst(char *buffer)
{
	ms_cudaMemcpyDst_t* ms = TEE_CAST(ms_cudaMemcpyDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;
	void* _tmp_src = ms->ms_src;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpyDst(_in_dst, (const void*)_tmp_src, _tmp_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _in_dst, (const void*)_tmp_src, _tmp_count, ms->ms_kind, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpySrcDst(char *buffer)
{
	ms_cudaMemcpySrcDst_t* ms = TEE_CAST(ms_cudaMemcpySrcDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpySrcDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + _tmp_count);
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpySrcDst(_in_dst, (const void*)_in_src, _tmp_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _in_dst, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyPeer(char *buffer)
{
	ms_cudaMemcpyPeer_t* ms = TEE_CAST(ms_cudaMemcpyPeer_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyPeer_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpyPeer(_tmp_dst, ms->ms_dstDevice, (const void*)_tmp_src, ms->ms_srcDevice, ms->ms_count);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dstDevice, (const void*)_tmp_src, ms->ms_srcDevice, ms->ms_count, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DNone(char *buffer)
{
	ms_cudaMemcpy2DNone_t* ms = TEE_CAST(ms_cudaMemcpy2DNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpy2DNone(_tmp_dst, ms->ms_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DSrc(char *buffer)
{
	ms_cudaMemcpy2DSrc_t* ms = TEE_CAST(ms_cudaMemcpy2DSrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DSrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _tmp_spitch = ms->ms_spitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_src = _tmp_height * _tmp_spitch;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpy2DSrc(_tmp_dst, ms->ms_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpy2DDst(char *buffer)
{
	ms_cudaMemcpy2DDst_t* ms = TEE_CAST(ms_cudaMemcpy2DDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_dpitch = ms->ms_dpitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_dst = _tmp_height * _tmp_dpitch;
	void* _in_dst = NULL;
	void* _tmp_src = ms->ms_src;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpy2DDst(_in_dst, _tmp_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, _tmp_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, _tmp_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpy2DSrcDst(char *buffer)
{
	ms_cudaMemcpy2DSrcDst_t* ms = TEE_CAST(ms_cudaMemcpy2DSrcDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DSrcDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_dpitch = ms->ms_dpitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_dst = _tmp_height * _tmp_dpitch;
	void* _in_dst = NULL;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + _tmp_height * _tmp_dpitch);
	size_t _tmp_spitch = ms->ms_spitch;
	size_t _len_src = _tmp_height * _tmp_spitch;
	void* _in_src = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpy2DSrcDst(_in_dst, _tmp_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, _tmp_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpy2DToArrayNone(char *buffer)
{
	ms_cudaMemcpy2DToArrayNone_t* ms = TEE_CAST(ms_cudaMemcpy2DToArrayNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DToArrayNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpy2DToArrayNone(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DToArraySrc(char *buffer)
{
	ms_cudaMemcpy2DToArraySrc_t* ms = TEE_CAST(ms_cudaMemcpy2DToArraySrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DToArraySrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_spitch = ms->ms_spitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_src = _tmp_height * _tmp_spitch;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpy2DToArraySrc(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpy2DFromArrayNone(char *buffer)
{
	ms_cudaMemcpy2DFromArrayNone_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DFromArrayNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;

	ms->ms_retval = cudaMemcpy2DFromArrayNone(_tmp_dst, ms->ms_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, ms->ms_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DFromArrayDst(char *buffer)
{
	ms_cudaMemcpy2DFromArrayDst_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DFromArrayDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_dpitch = ms->ms_dpitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_dst = _tmp_height * _tmp_dpitch;
	void* _in_dst = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpy2DFromArrayDst(_in_dst, _tmp_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, _tmp_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, _tmp_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpy2DArrayToArray(char *buffer)
{
	ms_cudaMemcpy2DArrayToArray_t* ms = TEE_CAST(ms_cudaMemcpy2DArrayToArray_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DArrayToArray_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaMemcpy2DArrayToArray(ms->ms_dst, ms->ms_wOffsetDst, ms->ms_hOffsetDst, ms->ms_src, ms->ms_wOffsetSrc, ms->ms_hOffsetSrc, ms->ms_width, ms->ms_height, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffsetDst, ms->ms_hOffsetDst, ms->ms_src, ms->ms_wOffsetSrc, ms->ms_hOffsetSrc, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpyToSymbolNone(char *buffer)
{
	ms_cudaMemcpyToSymbolNone_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToSymbolNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;
	void* _tmp_src = ms->ms_src;

	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaMemcpyToSymbolNone((const void*)_in_symbol, (const void*)_tmp_src, ms->ms_count, ms->ms_offset, ms->ms_kind);
	RPC_SERVER_DEBUG("(%s, %lx, %lx, %lx, %lx) => %lx", (const void*)_in_symbol, (const void*)_tmp_src, ms->ms_count, ms->ms_offset, ms->ms_kind, ms->ms_retval);
err:
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaMemcpyToSymbolSrc(char *buffer)
{
	ms_cudaMemcpyToSymbolSrc_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolSrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToSymbolSrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + strlen((const char*)_tmp_symbol) + 1);
	size_t _tmp_count = ms->ms_count;
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpyToSymbolSrc((const void*)_in_symbol, (const void*)_in_src, _tmp_count, ms->ms_offset, ms->ms_kind);
	RPC_SERVER_DEBUG("(%s, %lx, %lx, %lx, %lx) => %lx", (const void*)_in_symbol, (const void*)_in_src, _tmp_count, ms->ms_offset, ms->ms_kind, ms->ms_retval);
err:
	if (_in_symbol) free((void*)_in_symbol);
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyFromSymbolNone(char *buffer)
{
	ms_cudaMemcpyFromSymbolNone_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromSymbolNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;

	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaMemcpyFromSymbolNone(_tmp_dst, (const void*)_in_symbol, ms->ms_count, ms->ms_offset, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %s, %lx, %lx, %lx) => %lx", _tmp_dst, (const void*)_in_symbol, ms->ms_count, ms->ms_offset, ms->ms_kind, ms->ms_retval);
err:
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaMemcpyFromSymbolDst(char *buffer)
{
	ms_cudaMemcpyFromSymbolDst_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromSymbolDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0 + _tmp_count);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaMemcpyFromSymbolDst(_in_dst, (const void*)_in_symbol, _tmp_count, ms->ms_offset, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %s, %lx, %lx, %lx) => %lx", _in_dst, (const void*)_in_symbol, _tmp_count, ms->ms_offset, ms->ms_kind, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaMemcpyAsyncNone(char *buffer)
{
	ms_cudaMemcpyAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpyAsyncNone(_tmp_dst, (const void*)_tmp_src, ms->ms_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, (const void*)_tmp_src, ms->ms_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpyAsyncSrc(char *buffer)
{
	ms_cudaMemcpyAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpyAsyncSrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyAsyncSrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpyAsyncSrc(_tmp_dst, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyAsyncDst(char *buffer)
{
	ms_cudaMemcpyAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpyAsyncDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyAsyncDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;
	void* _tmp_src = ms->ms_src;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpyAsyncDst(_in_dst, (const void*)_tmp_src, _tmp_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, (const void*)_tmp_src, _tmp_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpyAsyncSrcDst(char *buffer)
{
	ms_cudaMemcpyAsyncSrcDst_t* ms = TEE_CAST(ms_cudaMemcpyAsyncSrcDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyAsyncSrcDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + _tmp_count);
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpyAsyncSrcDst(_in_dst, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyPeerAsync(char *buffer)
{
	ms_cudaMemcpyPeerAsync_t* ms = TEE_CAST(ms_cudaMemcpyPeerAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyPeerAsync_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpyPeerAsync(_tmp_dst, ms->ms_dstDevice, (const void*)_tmp_src, ms->ms_srcDevice, ms->ms_count, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dstDevice, (const void*)_tmp_src, ms->ms_srcDevice, ms->ms_count, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DAsyncNone(char *buffer)
{
	ms_cudaMemcpy2DAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpy2DAsyncNone(_tmp_dst, ms->ms_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DAsyncSrc(char *buffer)
{
	ms_cudaMemcpy2DAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncSrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DAsyncSrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _tmp_spitch = ms->ms_spitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_src = _tmp_height * _tmp_spitch;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpy2DAsyncSrc(_tmp_dst, ms->ms_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpy2DAsyncDst(char *buffer)
{
	ms_cudaMemcpy2DAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DAsyncDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_dpitch = ms->ms_dpitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_dst = _tmp_height * _tmp_dpitch;
	void* _in_dst = NULL;
	void* _tmp_src = ms->ms_src;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpy2DAsyncDst(_in_dst, _tmp_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, _tmp_dpitch, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpy2DAsyncSrcDst(char *buffer)
{
	ms_cudaMemcpy2DAsyncSrcDst_t* ms = TEE_CAST(ms_cudaMemcpy2DAsyncSrcDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DAsyncSrcDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_dpitch = ms->ms_dpitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_dst = _tmp_height * _tmp_dpitch;
	void* _in_dst = NULL;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + _tmp_height * _tmp_dpitch);
	size_t _tmp_spitch = ms->ms_spitch;
	size_t _len_src = _tmp_height * _tmp_spitch;
	void* _in_src = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpy2DAsyncSrcDst(_in_dst, _tmp_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, _tmp_dpitch, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpy2DToArrayAsyncNone(char *buffer)
{
	ms_cudaMemcpy2DToArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpy2DToArrayAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DToArrayAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpy2DToArrayAsyncNone(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_spitch, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DToArrayAsyncSrc(char *buffer)
{
	ms_cudaMemcpy2DToArrayAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpy2DToArrayAsyncSrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DToArrayAsyncSrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_spitch = ms->ms_spitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_src = _tmp_height * _tmp_spitch;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpy2DToArrayAsyncSrc(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_spitch, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpy2DFromArrayAsyncNone(char *buffer)
{
	ms_cudaMemcpy2DFromArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DFromArrayAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;

	ms->ms_retval = cudaMemcpy2DFromArrayAsyncNone(_tmp_dst, ms->ms_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, ms->ms_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpy2DFromArrayAsyncDst(char *buffer)
{
	ms_cudaMemcpy2DFromArrayAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpy2DFromArrayAsyncDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpy2DFromArrayAsyncDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_dpitch = ms->ms_dpitch;
	size_t _tmp_height = ms->ms_height;
	size_t _len_dst = _tmp_height * _tmp_dpitch;
	void* _in_dst = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpy2DFromArrayAsyncDst(_in_dst, _tmp_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, _tmp_dpitch, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_width, _tmp_height, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpyToSymbolAsyncNone(char *buffer)
{
	ms_cudaMemcpyToSymbolAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToSymbolAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;
	void* _tmp_src = ms->ms_src;

	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaMemcpyToSymbolAsyncNone((const void*)_in_symbol, (const void*)_tmp_src, ms->ms_count, ms->ms_offset, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%s, %lx, %lx, %lx, %lx, %lx) => %lx", (const void*)_in_symbol, (const void*)_tmp_src, ms->ms_count, ms->ms_offset, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaMemcpyToSymbolAsyncSrc(char *buffer)
{
	ms_cudaMemcpyToSymbolAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpyToSymbolAsyncSrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToSymbolAsyncSrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0 + strlen((const char*)_tmp_symbol) + 1);
	size_t _tmp_count = ms->ms_count;
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpyToSymbolAsyncSrc((const void*)_in_symbol, (const void*)_in_src, _tmp_count, ms->ms_offset, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%s, %lx, %lx, %lx, %lx, %lx) => %lx", (const void*)_in_symbol, (const void*)_in_src, _tmp_count, ms->ms_offset, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_symbol) free((void*)_in_symbol);
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyFromSymbolAsyncNone(char *buffer)
{
	ms_cudaMemcpyFromSymbolAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromSymbolAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0 + 0);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;

	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaMemcpyFromSymbolAsyncNone(_tmp_dst, (const void*)_in_symbol, ms->ms_count, ms->ms_offset, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %s, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, (const void*)_in_symbol, ms->ms_count, ms->ms_offset, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaMemcpyFromSymbolAsyncDst(char *buffer)
{
	ms_cudaMemcpyFromSymbolAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpyFromSymbolAsyncDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromSymbolAsyncDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0 + _tmp_count);
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaMemcpyFromSymbolAsyncDst(_in_dst, (const void*)_in_symbol, _tmp_count, ms->ms_offset, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %s, %lx, %lx, %lx, %lx) => %lx", _in_dst, (const void*)_in_symbol, _tmp_count, ms->ms_offset, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaMemset(char *buffer)
{
	ms_cudaMemset_t* ms = TEE_CAST(ms_cudaMemset_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemset_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaMemset(_tmp_devPtr, ms->ms_value, ms->ms_count);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", _tmp_devPtr, ms->ms_value, ms->ms_count, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemset2D(char *buffer)
{
	ms_cudaMemset2D_t* ms = TEE_CAST(ms_cudaMemset2D_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemset2D_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaMemset2D(_tmp_devPtr, ms->ms_pitch, ms->ms_value, ms->ms_width, ms->ms_height);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _tmp_devPtr, ms->ms_pitch, ms->ms_value, ms->ms_width, ms->ms_height, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemset3D(char *buffer)
{
	ms_cudaMemset3D_t* ms = TEE_CAST(ms_cudaMemset3D_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemset3D_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaMemset3D(ms->ms_pitchedDevPtr, ms->ms_value, ms->ms_extent);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx) => %lx", ms->ms_pitchedDevPtr, ms->ms_value, ms->ms_extent, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemsetAsync(char *buffer)
{
	ms_cudaMemsetAsync_t* ms = TEE_CAST(ms_cudaMemsetAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemsetAsync_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaMemsetAsync(_tmp_devPtr, ms->ms_value, ms->ms_count, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", _tmp_devPtr, ms->ms_value, ms->ms_count, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemset2DAsync(char *buffer)
{
	ms_cudaMemset2DAsync_t* ms = TEE_CAST(ms_cudaMemset2DAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemset2DAsync_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaMemset2DAsync(_tmp_devPtr, ms->ms_pitch, ms->ms_value, ms->ms_width, ms->ms_height, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_devPtr, ms->ms_pitch, ms->ms_value, ms->ms_width, ms->ms_height, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemset3DAsync(char *buffer)
{
	ms_cudaMemset3DAsync_t* ms = TEE_CAST(ms_cudaMemset3DAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemset3DAsync_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaMemset3DAsync(ms->ms_pitchedDevPtr, ms->ms_value, ms->ms_extent, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", ms->ms_pitchedDevPtr, ms->ms_value, ms->ms_extent, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaGetSymbolAddress(char *buffer)
{
	ms_cudaGetSymbolAddress_t* ms = TEE_CAST(ms_cudaGetSymbolAddress_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetSymbolAddress_t);

	TEE_Result status = TEE_SUCCESS;
	void** _tmp_devPtr = TEE_CAST(void**, buffer_start + 0);
	size_t _len_devPtr = 1 * sizeof(*_tmp_devPtr);
	void** _in_devPtr = NULL;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0 + 1 * sizeof(*_tmp_devPtr));
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;

	if (_tmp_devPtr != NULL) {
		if ((_in_devPtr = (void**)malloc(_len_devPtr)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_devPtr, 0, _len_devPtr);
	}
	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaGetSymbolAddress(_in_devPtr, (const void*)_in_symbol);
	RPC_SERVER_DEBUG("(%lx, %s) => %lx", _in_devPtr, (const void*)_in_symbol, ms->ms_retval);
err:
	if (_in_devPtr) {
		memcpy(_tmp_devPtr, _in_devPtr, _len_devPtr);
		free(_in_devPtr);
	}
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaGetSymbolSize(char *buffer)
{
	ms_cudaGetSymbolSize_t* ms = TEE_CAST(ms_cudaGetSymbolSize_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaGetSymbolSize_t);

	TEE_Result status = TEE_SUCCESS;
	size_t* _tmp_size = TEE_CAST(size_t*, buffer_start + 0);
	size_t _len_size = 1 * sizeof(*_tmp_size);
	size_t* _in_size = NULL;
	void* _tmp_symbol = TEE_CAST(void*, buffer_start + 0 + 1 * sizeof(*_tmp_size));
	size_t _len_symbol = _tmp_symbol ? strlen((const char*)(_tmp_symbol)) + 1 : 0;
	void* _in_symbol = NULL;

	if (_tmp_size != NULL) {
		if ((_in_size = (size_t*)malloc(_len_size)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_size, 0, _len_size);
	}
	if (_tmp_symbol != NULL) {
		_in_symbol = (void*)malloc(_len_symbol);
		if (_in_symbol == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_symbol, _tmp_symbol, _len_symbol);
		((char*)_in_symbol)[_len_symbol - 1] = '\0';
	}
	ms->ms_retval = cudaGetSymbolSize(_in_size, (const void*)_in_symbol);
	RPC_SERVER_DEBUG("(%lx, %s) => %lx", _in_size, (const void*)_in_symbol, ms->ms_retval);
err:
	if (_in_size) {
		memcpy(_tmp_size, _in_size, _len_size);
		free(_in_size);
	}
	if (_in_symbol) free((void*)_in_symbol);

	return status;
}

static TEE_Result tee_cudaMemPrefetchAsync(char *buffer)
{
	ms_cudaMemPrefetchAsync_t* ms = TEE_CAST(ms_cudaMemPrefetchAsync_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemPrefetchAsync_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaMemPrefetchAsync((const void*)_tmp_devPtr, ms->ms_count, ms->ms_dstDevice, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", (const void*)_tmp_devPtr, ms->ms_count, ms->ms_dstDevice, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemAdvise(char *buffer)
{
	ms_cudaMemAdvise_t* ms = TEE_CAST(ms_cudaMemAdvise_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemAdvise_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_devPtr = ms->ms_devPtr;

	ms->ms_retval = cudaMemAdvise((const void*)_tmp_devPtr, ms->ms_count, ms->ms_advice, ms->ms_device);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx) => %lx", (const void*)_tmp_devPtr, ms->ms_count, ms->ms_advice, ms->ms_device, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemRangeGetAttribute(char *buffer)
{
	ms_cudaMemRangeGetAttribute_t* ms = TEE_CAST(ms_cudaMemRangeGetAttribute_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemRangeGetAttribute_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_data = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_dataSize = ms->ms_dataSize;
	size_t _len_data = _tmp_dataSize;
	void* _in_data = NULL;
	void* _tmp_devPtr = ms->ms_devPtr;

	if (_tmp_data != NULL) {
		if ((_in_data = (void*)malloc(_len_data)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_data, 0, _len_data);
	}
	ms->ms_retval = cudaMemRangeGetAttribute(_in_data, _tmp_dataSize, ms->ms_attribute, (const void*)_tmp_devPtr, ms->ms_count);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx) => %lx", _in_data, _tmp_dataSize, ms->ms_attribute, (const void*)_tmp_devPtr, ms->ms_count, ms->ms_retval);
err:
	if (_in_data) {
		memcpy(_tmp_data, _in_data, _len_data);
		free(_in_data);
	}

	return status;
}

static TEE_Result tee_cudaMemcpyToArrayNone(char *buffer)
{
	ms_cudaMemcpyToArrayNone_t* ms = TEE_CAST(ms_cudaMemcpyToArrayNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToArrayNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpyToArrayNone(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_count, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpyToArraySrc(char *buffer)
{
	ms_cudaMemcpyToArraySrc_t* ms = TEE_CAST(ms_cudaMemcpyToArraySrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToArraySrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpyToArraySrc(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyFromArrayNone(char *buffer)
{
	ms_cudaMemcpyFromArrayNone_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromArrayNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;

	ms->ms_retval = cudaMemcpyFromArrayNone(_tmp_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_count, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpyFromArrayDst(char *buffer)
{
	ms_cudaMemcpyFromArrayDst_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromArrayDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpyFromArrayDst(_in_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, _tmp_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, _tmp_count, ms->ms_kind, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

static TEE_Result tee_cudaMemcpyArrayToArray(char *buffer)
{
	ms_cudaMemcpyArrayToArray_t* ms = TEE_CAST(ms_cudaMemcpyArrayToArray_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyArrayToArray_t);

	TEE_Result status = TEE_SUCCESS;

	ms->ms_retval = cudaMemcpyArrayToArray(ms->ms_dst, ms->ms_wOffsetDst, ms->ms_hOffsetDst, ms->ms_src, ms->ms_wOffsetSrc, ms->ms_hOffsetSrc, ms->ms_count, ms->ms_kind);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffsetDst, ms->ms_hOffsetDst, ms->ms_src, ms->ms_wOffsetSrc, ms->ms_hOffsetSrc, ms->ms_count, ms->ms_kind, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpyToArrayAsyncNone(char *buffer)
{
	ms_cudaMemcpyToArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyToArrayAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToArrayAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = ms->ms_src;

	ms->ms_retval = cudaMemcpyToArrayAsyncNone(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_tmp_src, ms->ms_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpyToArrayAsyncSrc(char *buffer)
{
	ms_cudaMemcpyToArrayAsyncSrc_t* ms = TEE_CAST(ms_cudaMemcpyToArrayAsyncSrc_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyToArrayAsyncSrc_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_src = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_src = _tmp_count;
	void* _in_src = NULL;

	if (_tmp_src != NULL) {
		_in_src = (void*)malloc(_len_src);
		if (_in_src == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memcpy((void*)_in_src, _tmp_src, _len_src);
	}
	ms->ms_retval = cudaMemcpyToArrayAsyncSrc(ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", ms->ms_dst, ms->ms_wOffset, ms->ms_hOffset, (const void*)_in_src, _tmp_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_src) free((void*)_in_src);

	return status;
}

static TEE_Result tee_cudaMemcpyFromArrayAsyncNone(char *buffer)
{
	ms_cudaMemcpyFromArrayAsyncNone_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayAsyncNone_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromArrayAsyncNone_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = ms->ms_dst;

	ms->ms_retval = cudaMemcpyFromArrayAsyncNone(_tmp_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _tmp_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, ms->ms_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);


	return status;
}

static TEE_Result tee_cudaMemcpyFromArrayAsyncDst(char *buffer)
{
	ms_cudaMemcpyFromArrayAsyncDst_t* ms = TEE_CAST(ms_cudaMemcpyFromArrayAsyncDst_t*, buffer);
	char* buffer_start = buffer + sizeof(ms_cudaMemcpyFromArrayAsyncDst_t);

	TEE_Result status = TEE_SUCCESS;
	void* _tmp_dst = TEE_CAST(void*, buffer_start + 0);
	size_t _tmp_count = ms->ms_count;
	size_t _len_dst = _tmp_count;
	void* _in_dst = NULL;

	if (_tmp_dst != NULL) {
		if ((_in_dst = (void*)malloc(_len_dst)) == NULL) {
			status = TEE_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_dst, 0, _len_dst);
	}
	ms->ms_retval = cudaMemcpyFromArrayAsyncDst(_in_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, _tmp_count, ms->ms_kind, ms->ms_stream);
	RPC_SERVER_DEBUG("(%lx, %lx, %lx, %lx, %lx, %lx, %lx) => %lx", _in_dst, ms->ms_src, ms->ms_wOffset, ms->ms_hOffset, _tmp_count, ms->ms_kind, ms->ms_stream, ms->ms_retval);
err:
	if (_in_dst) {
		memcpy(_tmp_dst, _in_dst, _len_dst);
		free(_in_dst);
	}

	return status;
}

const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv;} ecall_table[107];
} g_ecall_table = {
	107,
	{
		{(void*)(uintptr_t)tee_cudaLaunchKernelByName, 0},
		{(void*)(uintptr_t)tee_cudaThreadSynchronize, 0},
		{(void*)(uintptr_t)tee_cudaDeviceSynchronize, 0},
		{(void*)(uintptr_t)tee_cudaGetLastError, 0},
		{(void*)(uintptr_t)tee_cudaGetDeviceCount, 0},
		{(void*)(uintptr_t)tee_cudaGetDeviceProperties, 0},
		{(void*)(uintptr_t)tee_cudaDeviceGetAttribute, 0},
		{(void*)(uintptr_t)tee_cudaChooseDevice, 0},
		{(void*)(uintptr_t)tee_cudaSetDevice, 0},
		{(void*)(uintptr_t)tee_cudaGetDevice, 0},
		{(void*)(uintptr_t)tee_cudaSetValidDevices, 0},
		{(void*)(uintptr_t)tee_cudaSetDeviceFlags, 0},
		{(void*)(uintptr_t)tee_cudaGetDeviceFlags, 0},
		{(void*)(uintptr_t)tee_cudaStreamCreate, 0},
		{(void*)(uintptr_t)tee_cudaStreamCreateWithFlags, 0},
		{(void*)(uintptr_t)tee_cudaStreamCreateWithPriority, 0},
		{(void*)(uintptr_t)tee_cudaStreamGetPriority, 0},
		{(void*)(uintptr_t)tee_cudaStreamGetFlags, 0},
		{(void*)(uintptr_t)tee_cudaStreamDestroy, 0},
		{(void*)(uintptr_t)tee_cudaStreamWaitEvent, 0},
		{(void*)(uintptr_t)tee_cudaStreamSynchronize, 0},
		{(void*)(uintptr_t)tee_cudaStreamQuery, 0},
		{(void*)(uintptr_t)tee_cudaStreamAttachMemAsync, 0},
		{(void*)(uintptr_t)tee_cudaStreamBeginCapture, 0},
		{(void*)(uintptr_t)tee_cudaThreadExchangeStreamCaptureMode, 0},
		{(void*)(uintptr_t)tee_cudaStreamEndCapture, 0},
		{(void*)(uintptr_t)tee_cudaStreamIsCapturing, 0},
		{(void*)(uintptr_t)tee_cudaStreamGetCaptureInfo, 0},
		{(void*)(uintptr_t)tee_cudaMallocManaged, 0},
		{(void*)(uintptr_t)tee_cudaMalloc, 0},
		{(void*)(uintptr_t)tee_cudaMallocHost, 0},
		{(void*)(uintptr_t)tee_cudaMallocPitch, 0},
		{(void*)(uintptr_t)tee_cudaMallocArray, 0},
		{(void*)(uintptr_t)tee_cudaFree, 0},
		{(void*)(uintptr_t)tee_cudaFreeHost, 0},
		{(void*)(uintptr_t)tee_cudaFreeArray, 0},
		{(void*)(uintptr_t)tee_cudaFreeMipmappedArray, 0},
		{(void*)(uintptr_t)tee_cudaHostAlloc, 0},
		{(void*)(uintptr_t)tee_cudaHostRegister, 0},
		{(void*)(uintptr_t)tee_cudaHostUnregister, 0},
		{(void*)(uintptr_t)tee_cudaHostGetDevicePointer, 0},
		{(void*)(uintptr_t)tee_cudaHostGetFlags, 0},
		{(void*)(uintptr_t)tee_cudaMalloc3D, 0},
		{(void*)(uintptr_t)tee_cudaMalloc3DArray, 0},
		{(void*)(uintptr_t)tee_cudaMallocMipmappedArray, 0},
		{(void*)(uintptr_t)tee_cudaGetMipmappedArrayLevel, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy3D, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy3DPeer, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy3DAsync, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy3DPeerAsync, 0},
		{(void*)(uintptr_t)tee_cudaMemGetInfo, 0},
		{(void*)(uintptr_t)tee_cudaArrayGetInfo, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpySrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpySrcDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyPeer, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DSrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DSrcDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DToArrayNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DToArraySrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DFromArrayNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DFromArrayDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DArrayToArray, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToSymbolNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToSymbolSrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromSymbolNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromSymbolDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyAsyncSrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyAsyncDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyAsyncSrcDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyPeerAsync, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DAsyncSrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DAsyncDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DAsyncSrcDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DToArrayAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DToArrayAsyncSrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DFromArrayAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpy2DFromArrayAsyncDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToSymbolAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToSymbolAsyncSrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromSymbolAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromSymbolAsyncDst, 0},
		{(void*)(uintptr_t)tee_cudaMemset, 0},
		{(void*)(uintptr_t)tee_cudaMemset2D, 0},
		{(void*)(uintptr_t)tee_cudaMemset3D, 0},
		{(void*)(uintptr_t)tee_cudaMemsetAsync, 0},
		{(void*)(uintptr_t)tee_cudaMemset2DAsync, 0},
		{(void*)(uintptr_t)tee_cudaMemset3DAsync, 0},
		{(void*)(uintptr_t)tee_cudaGetSymbolAddress, 0},
		{(void*)(uintptr_t)tee_cudaGetSymbolSize, 0},
		{(void*)(uintptr_t)tee_cudaMemPrefetchAsync, 0},
		{(void*)(uintptr_t)tee_cudaMemAdvise, 0},
		{(void*)(uintptr_t)tee_cudaMemRangeGetAttribute, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToArrayNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToArraySrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromArrayNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromArrayDst, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyArrayToArray, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToArrayAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyToArrayAsyncSrc, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromArrayAsyncNone, 0},
		{(void*)(uintptr_t)tee_cudaMemcpyFromArrayAsyncDst, 0},
	}
};

int rpc_dispatch(char* buffer)
{
	uint32_t cmd_id = *(uint32_t*)buffer;
	ecall_invoke_entry entry = TEE_CAST(ecall_invoke_entry, g_ecall_table.ecall_table[cmd_id].ecall_addr);
	return (*entry)(buffer + sizeof(uint32_t));
}
