
#include <assert.h>
#include <stdio.h>

#include "cuda_runtime_api.h"
#include "cuda_runtime_u.h"
#include "cuda_runtime_header.h"
#include "FatBinary.h"

#define KERNEL_ARGC_SIZE 64

static uint32_t* parameter_cache[MAX_FUNCS];
static uint32_t parameter_cnt_cache[MAX_FUNCS];

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
        char *func_name = NULL;
        cudaError_t ret;
        uint32_t n_par;
        uint32_t *parameters;
        uint32_t total_parameter_sizes = 0;
        void* args_copy = NULL;
        int args_copy_offset = 0;
		int func_idx = 0;

        for (int i = 0;i < __cuda_runtime_func_cnt;i++) {
                if (__cuda_runtime_func_ptr[i] == func) {
                        func_name = __cuda_runtime_func_names[i];
						func_idx = i;
                }
        }
        if (!func_name) {
                return cudaErrorLaunchFailure;
        }

		if (!parameter_cache[func_idx]) {
			parameter_cache[func_idx] = (uint32_t*) malloc(sizeof(uint32_t) * 20);
			parameters = parameter_cache[func_idx];
			// ret = cudaFuncGetParametersByName(&n_par, parameters, func_name, strlen(func_name) + 1);

			// if (ret) {
			// 	return ret;
			// }
			auto func = fatbin_handle->get_function(func_name);
			if (func == NULL) {
				return cudaErrorLaunchFailure;
			}
			n_par = 0;
			for (auto &param : func->param_data) {
				parameters[func->param_data.size() - 1 - n_par] = param.size;
				n_par ++;
			}
			
			parameter_cnt_cache[func_idx] = n_par;
		} else {
			parameters = parameter_cache[func_idx];
			n_par = parameter_cnt_cache[func_idx];
		}

        for (int i = 0;i < n_par; i++)
                total_parameter_sizes += parameters[i];
        
        args_copy = malloc(total_parameter_sizes);

        for (int i = 0;i < n_par;i++) {
                memcpy((char*)args_copy + args_copy_offset, args[i], parameters[i]);
                args_copy_offset += parameters[i];
        }

        ret = cudaLaunchKernelByName(func_name, strlen(func_name) + 1, gridDim, blockDim, args_copy, total_parameter_sizes, parameters, sizeof(uint32_t) * n_par, sharedMem, stream);

        free(args_copy);

        return ret;
}

/*
cudaError_t  cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
	if (kind == cudaMemcpyHostToDevice) {
        return cudaMemcpyToDevice(dst, (void*) src, count);
    } else if (kind == cudaMemcpyDeviceToHost) {
        return cudaMemcpyToHost(dst, (void*) src, count);
    } else {
        return cudaErrorUnknown;
    }
}
*/

#define _CASE(x) case x: return #x;

const char*  cudaGetErrorString(cudaError_t error) {
	switch (error) {
	_CASE(cudaSuccess)
	_CASE(cudaErrorMissingConfiguration)
	_CASE(cudaErrorMemoryAllocation)
	_CASE(cudaErrorInitializationError)
	_CASE(cudaErrorLaunchFailure)
	_CASE(cudaErrorPriorLaunchFailure)
	_CASE(cudaErrorLaunchTimeout)
	_CASE(cudaErrorLaunchOutOfResources)
	_CASE(cudaErrorInvalidDeviceFunction)
	_CASE(cudaErrorInvalidConfiguration)
	_CASE(cudaErrorInvalidDevice)
	_CASE(cudaErrorInvalidValue)
	_CASE(cudaErrorInvalidPitchValue)
	_CASE(cudaErrorInvalidSymbol)
	_CASE(cudaErrorMapBufferObjectFailed)
	_CASE(cudaErrorUnmapBufferObjectFailed)
	_CASE(cudaErrorInvalidHostPointer)
	_CASE(cudaErrorInvalidDevicePointer)
	_CASE(cudaErrorInvalidTexture)
	_CASE(cudaErrorInvalidTextureBinding)
	_CASE(cudaErrorInvalidChannelDescriptor)
	_CASE(cudaErrorInvalidMemcpyDirection)
	_CASE(cudaErrorAddressOfConstant)
	_CASE(cudaErrorTextureFetchFailed)
	_CASE(cudaErrorTextureNotBound)
	_CASE(cudaErrorSynchronizationError)
	_CASE(cudaErrorInvalidFilterSetting)
	_CASE(cudaErrorInvalidNormSetting)
	_CASE(cudaErrorMixedDeviceExecution)
	_CASE(cudaErrorCudartUnloading)
	_CASE(cudaErrorUnknown)
	_CASE(cudaErrorNotYetImplemented)
	_CASE(cudaErrorMemoryValueTooLarge)
	_CASE(cudaErrorInvalidResourceHandle)
	_CASE(cudaErrorNotReady)
	_CASE(cudaErrorInsufficientDriver)
	_CASE(cudaErrorSetOnActiveProcess)
	_CASE(cudaErrorNoDevice)
	_CASE(cudaErrorStartupFailure)
	_CASE(cudaErrorApiFailureBase)
		default:
		break;
	}
	return "unimplemented";
}