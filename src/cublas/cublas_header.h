
#pragma once

#include "debug.h"
#include <iostream>

#define cublas_log_info(format, ...) log_info("CUBLAS: " format, ## __VA_ARGS__)
#define cublas_log_warn(format, ...) log_warn("CUBLAS: " format, ## __VA_ARGS__)
#define cublas_log_err(format, ...) log_err("CUBLAS: " format, ## __VA_ARGS__)
#define cublas_log_call() cublas_log_info("=> %s", __FUNCTION__)
#define cublas_exit() exit(1)
#define cublas_not_implemented(ret) cublas_log_warn("%s not implemented", __FUNCTION__); return ret
#define cublas_not_implemented_noreturn cublas_log_warn("%s not implemented", __FUNCTION__)
#define CUBLAS_NOT_IMPLEMENTED { cublas_not_implemented(CUBLAS_STATUS_NOT_SUPPORTED); }

#define NOT_IMPLEMENTED cublas_log_warn("%s not implemented", __FUNCTION__)
#define NOT_IMPLEMENTED_RET(x) cublas_log_warn("%s not implemented", __FUNCTION__); return x;