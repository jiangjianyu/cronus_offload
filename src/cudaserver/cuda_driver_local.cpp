
#include <cuda.h>
#include <stdio.h>

#include "cuda_table.h"

#define FORWARD_FUNC(func, decl, call, uuid, idx)                           \
    typedef CUresult (*func##_t)decl;                                     \
    extern "C" CUresult func decl {                                        \
        const void* vtable = NULL;                                          \
        auto r = cuGetExportTable(&vtable, (const CUuuid*)&uuid);   \
        func##_t f = (func##_t)((void**)vtable)[idx];                       \
        return (*f)call;                                                  \
    }

FORWARD_FUNC(internal_func_4_1, (int ptr1, void* ptr2, float* ptr3), (ptr1, ptr2, ptr3), unknown_guid, 1);
FORWARD_FUNC(cudart_interface_internal2, (CUcontext *pctx, CUdevice dev), (pctx, dev), cudart_interface_guid, 2);
FORWARD_FUNC(context_local_storage_get_state, (void** ctx_state, CUcontext cu_ctx, void* mgr), (ctx_state, cu_ctx, mgr), context_local_storage_interface_v0301_guid, 2);

extern "C" CUresult cuDevicePrimaryCtxRetainRemote(CUcontext *pctx, CUdevice dev) {
    return cuDevicePrimaryCtxRetain(pctx, dev);
}