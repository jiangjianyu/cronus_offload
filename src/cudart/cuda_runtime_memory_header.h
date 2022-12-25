
#pragma once

#define DEFINE_CUDART_MEMCPY_ALL(funcname,desc,call)    \
    cudaError_t funcname desc {                         \
        if (kind == cudaMemcpyDeviceToDevice) {         \
            return funcname ## None call;               \
        } else if (kind == cudaMemcpyHostToDevice) {    \
            return funcname ## Src call;                \
        } else if (kind == cudaMemcpyDeviceToHost) {    \
            return funcname ## Dst call;                \
        } else if (kind == cudaMemcpyHostToHost) {      \
            return funcname ## SrcDst call;             \
        } else {                                        \
            cudart_log_err("error memcpy kind");        \
            return cudaErrorInvalidValue;               \
        }                                               \
    }

#define DEFINE_CUDART_MEMCPY_SRC(funcname,desc,call)    \
    cudaError_t funcname desc {                         \
        if (kind == cudaMemcpyDeviceToDevice) {         \
            return funcname ## None call;               \
        } else if (kind == cudaMemcpyHostToDevice) {    \
            return funcname ## Src call;                \
        } else if (kind == cudaMemcpyDeviceToHost) {    \
            return funcname ## None call;               \
        } else if (kind == cudaMemcpyHostToHost) {      \
            return funcname ## Src call;                \
        } else {                                        \
            cudart_log_err("error memcpy kind");        \
            return cudaErrorInvalidValue;               \
        }                                               \
    }

#define DEFINE_CUDART_MEMCPY_DST(funcname,desc,call)    \
    cudaError_t funcname desc {                         \
        if (kind == cudaMemcpyDeviceToDevice) {         \
            return funcname ## None call;               \
        } else if (kind == cudaMemcpyHostToDevice) {    \
            return funcname ## None call;               \
        } else if (kind == cudaMemcpyDeviceToHost) {    \
            return funcname ## Dst call;                \
        } else if (kind == cudaMemcpyHostToHost) {      \
            return funcname ## Dst call;                \
        } else {                                        \
            cudart_log_err("error memcpy kind");        \
            return cudaErrorInvalidValue;               \
        }                                               \
    }