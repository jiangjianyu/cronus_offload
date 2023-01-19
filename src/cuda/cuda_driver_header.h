
#pragma once

#include "debug.h"
#include <iostream>

#define cudadrv_log_info(format, ...) log_info("CUDADRV: " format, ## __VA_ARGS__)
#define cudadrv_log_warn(format, ...) log_warn("CUDADRV: " format, ## __VA_ARGS__)
#define cudadrv_log_err(format, ...) log_err("CUDADRV: " format, ## __VA_ARGS__)
#define cudadrv_log_call() cudadrv_log_info("=> %s", __FUNCTION__)
#define cudadrv_exit() exit(1)
#define cudadrv_not_implemented(ret) cudadrv_log_warn("%s not implemented", __FUNCTION__); return ret
#define cudadrv_not_implemented_noreturn cudadrv_log_warn("%s not implemented", __FUNCTION__)
#define CUDADRV_NOT_IMPLEMENTED { cudadrv_not_implemented(CUDA_ERROR_NOT_SUPPORTED); }

#define NOT_IMPLEMENTED cudadrv_log_warn("%s not implemented", __FUNCTION__)
#define NOT_IMPLEMENTED_RET(x) cudadrv_log_warn("%s not implemented", __FUNCTION__); return x;

typedef void (*ctx_dtor_ct_t)(CUcontext, void*, void*);

typedef struct {
    void* mgr;
    void* ctx_state;
    ctx_dtor_ct_t dtor_cb;
} ctx_local_storage_t;

extern ctx_local_storage_t local_storage;