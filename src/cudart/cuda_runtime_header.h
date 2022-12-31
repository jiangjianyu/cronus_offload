
#pragma once

#include "FatBinary.h"
#include "debug.h"
#include <list>
#include <unordered_map>

// TODO: no std support
// currently we cannot use std in the module, as there will be abi compatability problem
#define MAX_FUNCS (8196 * 4)
#define MAX_CUBIN (512)
extern std::list<FatBinary*> *fatbins;
extern std::unordered_map<intptr_t, char*> *cuda_runtime_func;

#define cudart_log_info(format, ...) log_info("CUDART: " format, ## __VA_ARGS__)
#define cudart_log_warn(format, ...) log_warn("CUDART: " format, ## __VA_ARGS__)
#define cudart_log_err(format, ...) log_err("CUDART: " format, ## __VA_ARGS__)
#define cudart_log_call() cudart_log_info("=> %s", __FUNCTION__)
#define cudart_exit() exit(1)
#define cudart_not_implemented(ret) cudart_log_warn("%s not implemented", __FUNCTION__); return ret
#define cudart_not_implemented_noreturn cudart_log_warn("%s not implemented", __FUNCTION__)
#define CUDART_NOT_IMPLEMENTED { cudart_not_implemented(cudaErrorNotSupported); }

#define NOT_IMPLEMENTED cudart_log_warn("%s not implemented", __FUNCTION__)
#define NOT_IMPLEMENTED_RET(x) cudart_log_warn("%s not implemented", __FUNCTION__); return x;