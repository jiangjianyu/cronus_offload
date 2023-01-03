
#pragma once

#include "debug.h"

#define cudaserver_log_info(format, ...) log_info("CUDA: " format, ## __VA_ARGS__)
#define cudaserver_log_warn(format, ...) log_warn("CUDA: " format, ## __VA_ARGS__)
#define cudaserver_log_err(format, ...) log_err("CUDA: " format, ## __VA_ARGS__)
#define cudaserver_log_call() cudaserver_log_info("=> %s", __FUNCTION__)
#define cudart_exit() exit(1)
