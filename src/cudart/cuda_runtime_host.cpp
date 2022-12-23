
#include <assert.h>
#include <stdio.h>

#include "cuda_runtime_api.h"
#include "cuda_runtime_u.h"
#include "cuda_runtime_header.h"

dim3 pushed_gridDim;
dim3 pushed_blockDim;
size_t pushed_sharedMem;
struct CUstream_st *pushed_stream;
FatBinary *fatbin_handle = NULL;
extern "C" void init_rpc();

// for function registration
char* __cuda_runtime_func_names[MAX_FUNCS];
const char* __cuda_runtime_func_ptr[MAX_FUNCS];
int __cuda_runtime_func_cnt = 0;

extern "C" unsigned __cudaPushCallConfiguration(
        dim3 gridDim,
        dim3 blockDim,
        size_t sharedMem,
        struct CUstream_st *stream) {
	pushed_gridDim = gridDim;
	pushed_blockDim = blockDim;
	pushed_sharedMem = sharedMem;
	pushed_stream = stream;
	return cudaSuccess;
}

extern "C" cudaError_t __cudaPopCallConfiguration(
        dim3 *gridDim,
        dim3 *blockDim,
        size_t *sharedMem,
        void *stream
) {
	cudaStream_t *__stream = (cudaStream_t*)stream;
	*gridDim = pushed_gridDim;
	*blockDim = pushed_blockDim;
	*sharedMem = pushed_sharedMem;
	*__stream = (cudaStream_t)(long)pushed_stream;
	return cudaSuccess;
}

//////////////////////// from host_runtime.h ////////////////

extern "C" {
extern void** CUDARTAPI __cudaRegisterFatBinary(
  void *fatCubin
) {
    cudart_log_call();
    init_rpc();
    fatbin_handle = new FatBinary(fatCubin);
    fatbin_handle->parse();
    return (void**)&fatbin_handle;
}

extern void CUDARTAPI __cudaRegisterFatBinaryEnd(
  void **fatCubinHandle
) {
    cudart_log_call();
}

extern void CUDARTAPI __cudaUnregisterFatBinary(
  void **fatCubinHandle
) {
	cudart_log_call();
	rpc_close();
}

extern void CUDARTAPI __cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
) {
    NOT_IMPLEMENTED;
}

extern void CUDARTAPI __cudaRegisterManagedVar(
        void **fatCubinHandle,
        void **hostVarPtrAddress,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
) {
    NOT_IMPLEMENTED;
}

extern char CUDARTAPI __cudaInitModule(
        void **fatCubinHandle
) {
    cudart_log_call();
    return 0;
}

extern void CUDARTAPI __cudaRegisterTexture(
        void                    **fatCubinHandle,
  const struct textureReference  *hostVar,
  const void                    **deviceAddress,
  const char                     *deviceName,
        int                       dim,       
        int                       norm,      
        int                        ext        
);

extern void CUDARTAPI __cudaRegisterSurface(
        void                    **fatCubinHandle,
  const struct surfaceReference  *hostVar,
  const void                    **deviceAddress,
  const char                     *deviceName,
        int                       dim,       
        int                       ext        
) {
    NOT_IMPLEMENTED;
}

extern void CUDARTAPI __cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
) {
    if (__cuda_runtime_func_cnt >= MAX_FUNCS) {
        cudart_log_err("OOM of func names: [%d == %d]", __cuda_runtime_func_cnt, MAX_FUNCS);
        cudart_exit();
    }
    __cuda_runtime_func_names[__cuda_runtime_func_cnt] = deviceFun;
    __cuda_runtime_func_ptr[__cuda_runtime_func_cnt] = hostFun;
    __cuda_runtime_func_cnt += 1;
}

}