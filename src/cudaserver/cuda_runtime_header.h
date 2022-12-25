
#pragma once

#include "debug.h"

#define cudart_log_info(format, ...) log_info("CUDART: " format, ## __VA_ARGS__)
#define cudart_log_warn(format, ...) log_warn("CUDART: " format, ## __VA_ARGS__)
#define cudart_log_err(format, ...) log_err("CUDART: " format, ## __VA_ARGS__)
#define cudart_log_call() cudart_log_info("=> %s", __FUNCTION__)
#define cudart_exit() exit(1)
