#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "cuda.h"
#include "cuda_driver_header.h"
#include "cuda_driver_u.h"
#include "cuda_table.h"

#define _CASE(x) case x: *pStr = #x; break;

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
    return cuGetErrorName(error, pStr);
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
    switch (error) {
        _CASE(CUDA_SUCCESS)
        _CASE(CUDA_ERROR_INVALID_VALUE)
        _CASE(CUDA_ERROR_OUT_OF_MEMORY)
        _CASE(CUDA_ERROR_NOT_INITIALIZED)
        _CASE(CUDA_ERROR_DEINITIALIZED)
        _CASE(CUDA_ERROR_PROFILER_DISABLED)
        _CASE(CUDA_ERROR_PROFILER_NOT_INITIALIZED)
        _CASE(CUDA_ERROR_PROFILER_ALREADY_STARTED)
        _CASE(CUDA_ERROR_PROFILER_ALREADY_STOPPED)
        _CASE(CUDA_ERROR_STUB_LIBRARY)
        _CASE(CUDA_ERROR_NO_DEVICE)
        _CASE(CUDA_ERROR_INVALID_DEVICE)
        _CASE(CUDA_ERROR_DEVICE_NOT_LICENSED)
        _CASE(CUDA_ERROR_INVALID_IMAGE)
        _CASE(CUDA_ERROR_INVALID_CONTEXT)
        _CASE(CUDA_ERROR_CONTEXT_ALREADY_CURRENT)
        _CASE(CUDA_ERROR_MAP_FAILED)
        _CASE(CUDA_ERROR_UNMAP_FAILED)
        _CASE(CUDA_ERROR_ARRAY_IS_MAPPED)
        _CASE(CUDA_ERROR_ALREADY_MAPPED)
        _CASE(CUDA_ERROR_NO_BINARY_FOR_GPU)
        _CASE(CUDA_ERROR_ALREADY_ACQUIRED)
        _CASE(CUDA_ERROR_NOT_MAPPED)
        _CASE(CUDA_ERROR_NOT_MAPPED_AS_ARRAY)
        _CASE(CUDA_ERROR_NOT_MAPPED_AS_POINTER)
        _CASE(CUDA_ERROR_ECC_UNCORRECTABLE)
        _CASE(CUDA_ERROR_UNSUPPORTED_LIMIT)
        _CASE(CUDA_ERROR_CONTEXT_ALREADY_IN_USE)
        _CASE(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)
        _CASE(CUDA_ERROR_INVALID_PTX)
        _CASE(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT)
        _CASE(CUDA_ERROR_NVLINK_UNCORRECTABLE)
        _CASE(CUDA_ERROR_JIT_COMPILER_NOT_FOUND)
        _CASE(CUDA_ERROR_JIT_COMPILATION_DISABLED)
        _CASE(CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY)
        _CASE(CUDA_ERROR_INVALID_SOURCE)
        _CASE(CUDA_ERROR_FILE_NOT_FOUND)
        _CASE(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND)
        _CASE(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED)
        _CASE(CUDA_ERROR_OPERATING_SYSTEM)
        _CASE(CUDA_ERROR_INVALID_HANDLE)
        _CASE(CUDA_ERROR_ILLEGAL_STATE)
        _CASE(CUDA_ERROR_NOT_FOUND)
        _CASE(CUDA_ERROR_NOT_READY)
        _CASE(CUDA_ERROR_ILLEGAL_ADDRESS)
        _CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
        _CASE(CUDA_ERROR_LAUNCH_TIMEOUT)
        _CASE(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING)
        _CASE(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
        _CASE(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
        _CASE(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE)
        _CASE(CUDA_ERROR_CONTEXT_IS_DESTROYED)
        _CASE(CUDA_ERROR_ASSERT)
        _CASE(CUDA_ERROR_TOO_MANY_PEERS)
        _CASE(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
        _CASE(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
        _CASE(CUDA_ERROR_HARDWARE_STACK_ERROR)
        _CASE(CUDA_ERROR_ILLEGAL_INSTRUCTION)
        _CASE(CUDA_ERROR_MISALIGNED_ADDRESS)
        _CASE(CUDA_ERROR_INVALID_ADDRESS_SPACE)
        _CASE(CUDA_ERROR_INVALID_PC)
        _CASE(CUDA_ERROR_LAUNCH_FAILED)
        _CASE(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE)
        _CASE(CUDA_ERROR_NOT_PERMITTED)
        _CASE(CUDA_ERROR_NOT_SUPPORTED)
        _CASE(CUDA_ERROR_SYSTEM_NOT_READY)
        _CASE(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH)
        _CASE(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)
        _CASE(CUDA_ERROR_MPS_CONNECTION_FAILED)
        _CASE(CUDA_ERROR_MPS_RPC_FAILURE)
        _CASE(CUDA_ERROR_MPS_SERVER_NOT_READY)
        _CASE(CUDA_ERROR_MPS_MAX_CLIENTS_REACHED)
        _CASE(CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_MERGE)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_UNMATCHED)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_UNJOINED)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_ISOLATION)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT)
        _CASE(CUDA_ERROR_CAPTURED_EVENT)
        _CASE(CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD)
        _CASE(CUDA_ERROR_TIMEOUT)
        _CASE(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE)
        _CASE(CUDA_ERROR_EXTERNAL_DEVICE)
        _CASE(CUDA_ERROR_UNKNOWN)
        default:*pStr = NULL; return CUDA_ERROR_INVALID_VALUE;
	}
	return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
    *driverVersion = CUDA_VERSION;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
    auto handle = dlopen(NULL, RTLD_LAZY);

    auto dl_addr = dlsym(handle, symbol);
    if (dl_addr != NULL) {
        goto success;
    }

    char buf[50];
    for (int i = 1;i <= 3;i++) {
        sprintf(buf, "%s_v%d", symbol, i);
        dl_addr = dlsym(handle, buf);
        if (dl_addr != NULL) {
            goto success;
        }
    }
    
    dlclose(handle);
    cudadrv_log_err("not implemented %s %lx", symbol, flags);
    return CUDA_ERROR_NOT_SUPPORTED;
success:
    *pfn = dl_addr;
    return CUDA_SUCCESS;
}

typedef __uint128_t uuid_128_t;

static CUresult internal_dummy() {
    cudadrv_log_err("%s", __FUNCTION__); 
    return CUDA_SUCCESS;
}

static CUresult internal_2(CUcontext *pctx, CUdevice dev) {
    cudadrv_log_err("%s (%lx, %lx)", __FUNCTION__, (unsigned long int)pctx, (unsigned long int)dev); 
    return CUDA_SUCCESS;
}
static CUresult internal_7(uint64_t idx) {
    cudadrv_log_err("%s (%lx)", __FUNCTION__, (unsigned long int)idx);
    return CUDA_SUCCESS;
}

const intptr_t dummy_cu_module = 0x30f3;

static CUresult get_module_from_cubin(CUmodule *pmod, const void* fatbin_header, void* ptr1, void* ptr2) {
    cudadrv_log_warn("%s (%lx, %lx, %lx, %lx)", __FUNCTION__, (unsigned long int)pmod, (unsigned long int)fatbin_header, (unsigned long int)ptr1, (unsigned long int)ptr2);
    *pmod = (CUmodule)dummy_cu_module;
    return CUDA_SUCCESS;
}

struct CUfunc_st {
    const char *name;
};

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    // cudadrv_log_warn("%s (%lx, %lx, %s)", __FUNCTION__, hfunc, hmod, name);
    *hfunc = new CUfunc_st();
    (*hfunc)->name = name;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    cudadrv_log_warn("%s (%lx, %lx, %lx, %s)", __FUNCTION__, (unsigned long int)dptr, (unsigned long int)bytes, (unsigned long int)hmod, name);
    *dptr = (CUdeviceptr)name;
    if (bytes)
        *bytes = 1024;
    return CUDA_SUCCESS;
}

/**
 * Hidden function table from cuda driver API
 * reversed from test/nits/test_cuda && cublas
 */
static const void* cudart_interface_vtable[] = {
    (const void*)(10 * sizeof(const void*)),
    (const void*)&internal_dummy,
    (const void*)&cudart_interface_internal2,
    (const void*)&internal_dummy,
    (const void*)&internal_dummy,
    (const void*)&internal_dummy,
    (const void*)&get_module_from_cubin,
    (const void*)&internal_7,
    (const void*)&internal_dummy,
    (const void*)&internal_dummy,
    (const void*)NULL,
};

/**
 * Hidden function exported from cuda driver API
 * this function is reversed enginerred by the binary code
 * see test/nits/test_cuda.cpp
 */
static char callback_buffer_fn1[512];
static void* runtime_callback_hooks_fn1(void** callback_ptr, size_t *size) {
    *callback_ptr = (void*)callback_buffer_fn1;
    *size = 512;
    return *callback_ptr;
}

static char callback_buffer_fn5[0xe];
static void* runtime_callback_hooks_fn5(void** callback_ptr, size_t *size) {
    *callback_ptr = (void*)callback_buffer_fn5;
    *size = 0xe;
    return *callback_ptr;
}

/**
 * Hidden function exported from cuda driver API
 * this function is reversed enginerred by the binary code
 * see test/nits/test_cuda.cpp
 */
static const void* runtime_callback_hooks_vtable[] = {
    (const void*)(7 * sizeof(const void*)),
    (const void*)&internal_dummy,
    (const void*)&runtime_callback_hooks_fn1,
    (const void*)&internal_dummy,
    (const void*)&internal_dummy,
    (const void*)&internal_dummy,
    (const void*)&runtime_callback_hooks_fn5,
    (const void*)NULL,
};

static const void* tls_vtable[] = {
    (const void*)(1 * sizeof(const void*)),
};

ctx_local_storage_t local_storage = {
    .mgr = NULL,
    .ctx_state = NULL,
    .dtor_cb = (ctx_dtor_ct_t)NULL
};

static CUresult context_local_storage_ctor(CUcontext cu_ctx, void* mgr, void* ctx_state, void* dtor_cb) {
    cudadrv_log_warn("%s (%lx, %lx, %lx, %lx)", __FUNCTION__, (unsigned long int)cu_ctx, (unsigned long int)mgr, (unsigned long int)ctx_state, (unsigned long int)dtor_cb);
    local_storage.mgr = mgr;
    local_storage.ctx_state = ctx_state;
    local_storage.dtor_cb = (ctx_dtor_ct_t)dtor_cb;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (local_storage.mgr) {
        (*local_storage.dtor_cb)(*pctx, local_storage.mgr, local_storage.ctx_state);
    }
    return cuDevicePrimaryCtxRetainRemote(pctx, dev);
}

static uint32_t context_local_storage_dtor(size_t* s, void* ptr) {
    cudadrv_log_err("%s (%lx, %lx)", __FUNCTION__, (unsigned long int)s, (unsigned long int)ptr);
    return 0;
}


static CUresult context_local_storage_get_state(void** ctx_state, CUcontext cu_ctx, void* mgr) {
    cudadrv_log_warn("%s (%lx, %lx, %lx)", __FUNCTION__, (unsigned long int)ctx_state, (unsigned long int)cu_ctx, (unsigned long int)mgr);
    *ctx_state = local_storage.mgr;
    return CUDA_SUCCESS;
}

static const void* context_local_storage_interface_v0301_vtable[] = {
    (const void*)&context_local_storage_ctor,
    (const void*)&context_local_storage_dtor,
    (const void*)&context_local_storage_get_state,
    (const void*)(NULL),
};

static CUresult internal_dummy2(int ptr1, void* ptr2, float* ptr3) {
    cudadrv_log_err("%s %lx %lx %lx %f", __FUNCTION__, (unsigned long int)ptr1,(unsigned long int) ptr2, (unsigned long int)ptr3, *ptr3); 
    *ptr3 = 1;
    return CUDA_SUCCESS;
}

static const void* unknown_vtable[] = {
    (const void*)(2 * sizeof(void*)),
    (const void*)&internal_func_4_1,
    (const void*)&internal_dummy,
    (const void*)NULL
};

const struct ExportTable {
    const uuid_128_t* id;
    const void* vtable;
} cuda_exported_table [] = {
    {.id = (const uuid_128_t*)&cudart_interface_guid, .vtable = &cudart_interface_vtable[0]},
    {.id = (const uuid_128_t*)&runtime_callback_hooks_guid, .vtable = &runtime_callback_hooks_vtable[0]},
    {.id = (const uuid_128_t*)&tls_guid, .vtable = &tls_vtable[0]},
    {.id = (const uuid_128_t*)&context_local_storage_interface_v0301_guid, .vtable = &context_local_storage_interface_v0301_vtable[0]},
    {.id = (const uuid_128_t*)&unknown_guid, .vtable = &unknown_vtable[0]},
};

// see https://github.com/vosen/ZLUDA/blob/master/zluda/src/impl/export_table.rs
CUresult CUDAAPI cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
    const uuid_128_t* pExportTableId_128 = (const uuid_128_t*)pExportTableId;
    for (auto &table_entry: cuda_exported_table) {
        if (*pExportTableId_128 == *table_entry.id) {
            cudadrv_log_warn("error-prone %s impl: %lx %lx", __FUNCTION__, (unsigned long int)*pExportTableId_128, *((uint64_t*)pExportTableId_128 + 1));
            *ppExportTable = table_entry.vtable;
            return CUDA_SUCCESS;
        }
    }
    cudadrv_log_err("not implemented %lx %lx", (unsigned long int)*pExportTableId_128, *((uint64_t*)pExportTableId_128 + 1));
    *ppExportTable = (const void*) NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}