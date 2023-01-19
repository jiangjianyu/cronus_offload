
#include <cublas_v2.h>
#include <cublas_api.h>
#include "cublas_header.h"

#define CUBLAS_API_H_

#ifndef CUBLASWINAPI
#ifdef _WIN32
#define CUBLASWINAPI __stdcall
#else
#define CUBLASWINAPI
#endif
#endif

#include "driver_types.h"
#include "cuComplex.h" /* import complex data type */

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "library_types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#undef CUBLASAPI
#ifdef __CUDACC__
#define CUBLASAPI __attribute__((weak)) __host__ __device__
#else
#define CUBLASAPI __attribute__((weak))
#endif

#define WEAKAPI __attribute__((weak))

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle, int* version) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetProperty(libraryPropertyType type, int* value) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI size_t CUBLASWINAPI cublasGetCudartVersion(void) { return CUBLAS_VERSION; };

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetWorkspace_v2(cublasHandle_t handle,
                                                            void* workspace,
                                                            size_t workspaceSizeInBytes) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetSmCountTarget(cublasHandle_t handle, int* smCountTarget) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI const char* CUBLASWINAPI cublasGetStatusName(cublasStatus_t status) {
    #define _CASE(s) case s: return #s;
    switch (status) {
        _CASE(CUBLAS_STATUS_SUCCESS)
        _CASE(CUBLAS_STATUS_NOT_INITIALIZED)
        _CASE(CUBLAS_STATUS_ALLOC_FAILED)
        _CASE(CUBLAS_STATUS_INVALID_VALUE)
        _CASE(CUBLAS_STATUS_ARCH_MISMATCH)
        _CASE(CUBLAS_STATUS_MAPPING_ERROR)
        _CASE(CUBLAS_STATUS_EXECUTION_FAILED)
        _CASE(CUBLAS_STATUS_INTERNAL_ERROR)
        _CASE(CUBLAS_STATUS_NOT_SUPPORTED)
        _CASE(CUBLAS_STATUS_LICENSE_ERROR)
        default: return NULL;
    }
};

CUBLASAPI const char* CUBLASWINAPI cublasGetStatusString(cublasStatus_t status) { return cublasGetStatusName(status); }

/* Cublas logging */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLoggerConfigure(int logIsOn,
                                                            int logToStdOut,
                                                            int logToStdErr,
                                                            const char* logFileName) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetLoggerCallback(cublasLogCallback userCallback) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetLoggerCallback(cublasLogCallback* userCallback) CUBLAS_NOT_IMPLEMENTED;


/* Cublas vector/matrix */

WEAKAPI cublasStatus_t CUBLASWINAPI cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy) CUBLAS_NOT_IMPLEMENTED;
WEAKAPI cublasStatus_t CUBLASWINAPI cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) CUBLAS_NOT_IMPLEMENTED;

WEAKAPI cublasStatus_t CUBLASWINAPI cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) CUBLAS_NOT_IMPLEMENTED;
WEAKAPI cublasStatus_t CUBLASWINAPI cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) CUBLAS_NOT_IMPLEMENTED;

WEAKAPI cublasStatus_t CUBLASWINAPI cublasSetVectorAsync(
    int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream) CUBLAS_NOT_IMPLEMENTED;
WEAKAPI cublasStatus_t CUBLASWINAPI cublasGetVectorAsync(
    int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream) CUBLAS_NOT_IMPLEMENTED;

WEAKAPI cublasStatus_t CUBLASWINAPI
cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) CUBLAS_NOT_IMPLEMENTED;
WEAKAPI cublasStatus_t CUBLASWINAPI
cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI void CUBLASWINAPI cublasXerbla(const char* srName, int info) { cublas_not_implemented_noreturn; };


/* ---------------- CUBLAS BLAS1 functions ---------------- */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* result,
                                                   cudaDataType resultType,
                                                   cudaDataType executionType) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDnrm2_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx(cublasHandle_t handle,
                                                  int n,
                                                  const void* x,
                                                  cudaDataType xType,
                                                  int incx,
                                                  const void* y,
                                                  cudaDataType yType,
                                                  int incy,
                                                  void* result,
                                                  cudaDataType resultType,
                                                  cudaDataType executionType) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   const void* y,
                                                   cudaDataType yType,
                                                   int incy,
                                                   void* result,
                                                   cudaDataType resultType,
                                                   cudaDataType executionType) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2(cublasHandle_t handle,
                                                    int n,
                                                    const float* x,
                                                    int incx,
                                                    const float* y,
                                                    int incy,
                                                    float* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2(cublasHandle_t handle,
                                                    int n,
                                                    const double* x,
                                                    int incx,
                                                    const double* y,
                                                    int incy,
                                                    double* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* alpha, /* host or device pointer */
                                                   cudaDataType alphaType,
                                                   void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   cudaDataType executionType) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     float* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     double* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     cuComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle,
                                                      int n,
                                                      const float* alpha, /* host or device pointer */
                                                      cuComplex* x,
                                                      int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     cuDoubleComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle,
                                                      int n,
                                                      const double* alpha, /* host or device pointer */
                                                      cuDoubleComplex* x,
                                                      int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* alpha, /* host or device pointer */
                                                   cudaDataType alphaType,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* y,
                                                   cudaDataType yType,
                                                   int incy,
                                                   cudaDataType executiontype) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     float* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     double* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     cuComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     cuDoubleComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCopyEx(
    cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDcopy_v2(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDswap_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCswap_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSwapEx(
    cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIsamax_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIdamax_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIamaxEx(
    cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result /* host or device pointer */
) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIsamin_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIdamin_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIaminEx(
    cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result /* host or device pointer */
) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAsumEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* result,
                                                   cudaDataType resultType, /* host or device pointer */
                                                   cudaDataType executiontype) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDasum_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    float* x,
                                                    int incx,
                                                    float* y,
                                                    int incy,
                                                    const float* c,  /* host or device pointer */
                                                    const float* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    double* x,
                                                    int incx,
                                                    double* y,
                                                    int incy,
                                                    const double* c,  /* host or device pointer */
                                                    const double* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    cuComplex* x,
                                                    int incx,
                                                    cuComplex* y,
                                                    int incy,
                                                    const float* c,      /* host or device pointer */
                                                    const cuComplex* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle,
                                                     int n,
                                                     cuComplex* x,
                                                     int incx,
                                                     cuComplex* y,
                                                     int incy,
                                                     const float* c,  /* host or device pointer */
                                                     const float* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* y,
                                                    int incy,
                                                    const double* c,           /* host or device pointer */
                                                    const cuDoubleComplex* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle,
                                                     int n,
                                                     cuDoubleComplex* x,
                                                     int incx,
                                                     cuDoubleComplex* y,
                                                     int incy,
                                                     const double* c,  /* host or device pointer */
                                                     const double* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx(cublasHandle_t handle,
                                                  int n,
                                                  void* x,
                                                  cudaDataType xType,
                                                  int incx,
                                                  void* y,
                                                  cudaDataType yType,
                                                  int incy,
                                                  const void* c, /* host or device pointer */
                                                  const void* s,
                                                  cudaDataType csType,
                                                  cudaDataType executiontype) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle,
                                                     float* a,  /* host or device pointer */
                                                     float* b,  /* host or device pointer */
                                                     float* c,  /* host or device pointer */
                                                     float* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle,
                                                     double* a,  /* host or device pointer */
                                                     double* b,  /* host or device pointer */
                                                     double* c,  /* host or device pointer */
                                                     double* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle,
                                                     cuComplex* a,  /* host or device pointer */
                                                     cuComplex* b,  /* host or device pointer */
                                                     float* c,      /* host or device pointer */
                                                     cuComplex* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle,
                                                     cuDoubleComplex* a,  /* host or device pointer */
                                                     cuDoubleComplex* b,  /* host or device pointer */
                                                     double* c,           /* host or device pointer */
                                                     cuDoubleComplex* s) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotgEx(cublasHandle_t handle,
                                                   void* a, /* host or device pointer */
                                                   void* b, /* host or device pointer */
                                                   cudaDataType abType,
                                                   void* c, /* host or device pointer */
                                                   void* s, /* host or device pointer */
                                                   cudaDataType csType,
                                                   cudaDataType executiontype) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle,
                                                     int n,
                                                     float* x,
                                                     int incx,
                                                     float* y,
                                                     int incy,
                                                     const float* param) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle,
                                                     int n,
                                                     double* x,
                                                     int incx,
                                                     double* y,
                                                     int incy,
                                                     const double* param) CUBLAS_NOT_IMPLEMENTED; /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmEx(cublasHandle_t handle,
                                                   int n,
                                                   void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* y,
                                                   cudaDataType yType,
                                                   int incy,
                                                   const void* param, /* host or device pointer */
                                                   cudaDataType paramType,
                                                   cudaDataType executiontype) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle,
                                                      float* d1,       /* host or device pointer */
                                                      float* d2,       /* host or device pointer */
                                                      float* x1,       /* host or device pointer */
                                                      const float* y1, /* host or device pointer */
                                                      float* param) CUBLAS_NOT_IMPLEMENTED;   /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle,
                                                      double* d1,       /* host or device pointer */
                                                      double* d2,       /* host or device pointer */
                                                      double* x1,       /* host or device pointer */
                                                      const double* y1, /* host or device pointer */
                                                      double* param) CUBLAS_NOT_IMPLEMENTED;   /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmgEx(cublasHandle_t handle,
                                                    void* d1, /* host or device pointer */
                                                    cudaDataType d1Type,
                                                    void* d2, /* host or device pointer */
                                                    cudaDataType d2Type,
                                                    void* x1, /* host or device pointer */
                                                    cudaDataType x1Type,
                                                    const void* y1, /* host or device pointer */
                                                    cudaDataType y1Type,
                                                    void* param, /* host or device pointer */
                                                    cudaDataType paramType,
                                                    cudaDataType executiontype) CUBLAS_NOT_IMPLEMENTED;
/* --------------- CUBLAS BLAS2 functions  ---------------- */

/* GEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;
/* GBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

/* TRMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

/* TBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

/* TPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* AP,
                                                     float* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* AP,
                                                     double* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* AP,
                                                     cuComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* AP,
                                                     cuDoubleComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

/* TRSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

/* TPSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* AP,
                                                     float* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* AP,
                                                     double* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* AP,
                                                     cuComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* AP,
                                                     cuDoubleComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;
/* TBSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx) CUBLAS_NOT_IMPLEMENTED;

/* SYMV/HEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

/* SBMV/HBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

/* SPMV/HPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* AP,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* AP,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* AP,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* AP,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy) CUBLAS_NOT_IMPLEMENTED;

/* GER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2(cublasHandle_t handle,
                                                    int m,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    const float* y,
                                                    int incy,
                                                    float* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2(cublasHandle_t handle,
                                                    int m,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    const double* y,
                                                    int incy,
                                                    double* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

/* SYR/HER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    float* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    double* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const cuDoubleComplex* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* A,
                                                    int lda) CUBLAS_NOT_IMPLEMENTED;

/* SPR/HPR */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    float* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    double* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* AP) CUBLAS_NOT_IMPLEMENTED;

/* SYR2/HER2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     const float* y,
                                                     int incy,
                                                     float* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     const double* y,
                                                     int incy,
                                                     double* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda) CUBLAS_NOT_IMPLEMENTED;

/* SPR2/HPR2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     const float* y,
                                                     int incy,
                                                     float* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     const double* y,
                                                     int incy,
                                                     double* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* AP) CUBLAS_NOT_IMPLEMENTED;

/* ---------------- CUBLAS BLAS3 functions ---------------- */

/* GEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3m(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const cuComplex* A,
                                                    int lda,
                                                    const cuComplex* B,
                                                    int ldb,
                                                    const cuComplex* beta, /* host or device pointer */
                                                    cuComplex* C,
                                                    int ldc) CUBLAS_NOT_IMPLEMENTED;
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mEx(cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha,
                                                      const void* A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void* B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      const cuComplex* beta,
                                                      void* C,
                                                      cudaDataType Ctype,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm3m(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* A,
                                                    int lda,
                                                    const cuDoubleComplex* B,
                                                    int ldb,
                                                    const cuDoubleComplex* beta, /* host or device pointer */
                                                    cuDoubleComplex* C,
                                                    int ldc) CUBLAS_NOT_IMPLEMENTED;

#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  int k,
                                                  const __half* alpha, /* host or device pointer */
                                                  const __half* A,
                                                  int lda,
                                                  const __half* B,
                                                  int ldb,
                                                  const __half* beta, /* host or device pointer */
                                                  __half* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;
#endif
/* IO in FP16/FP32, computation in float */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmEx(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const float* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const void* B,
                                                    cudaDataType Btype,
                                                    int ldb,
                                                    const float* beta, /* host or device pointer */
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle,
                                                   cublasOperation_t transa,
                                                   cublasOperation_t transb,
                                                   int m,
                                                   int n,
                                                   int k,
                                                   const void* alpha, /* host or device pointer */
                                                   const void* A,
                                                   cudaDataType Atype,
                                                   int lda,
                                                   const void* B,
                                                   cudaDataType Btype,
                                                   int ldb,
                                                   const void* beta, /* host or device pointer */
                                                   void* C,
                                                   cudaDataType Ctype,
                                                   int ldc,
                                                   cublasComputeType_t computeType,
                                                   cublasGemmAlgo_t algo) CUBLAS_NOT_IMPLEMENTED;

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmEx(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha,
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const void* B,
                                                    cudaDataType Btype,
                                                    int ldb,
                                                    const cuComplex* beta,
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasUint8gemmBias(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          cublasOperation_t transc,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const unsigned char* A,
                                                          int A_bias,
                                                          int lda,
                                                          const unsigned char* B,
                                                          int B_bias,
                                                          int ldb,
                                                          unsigned char* C,
                                                          int C_bias,
                                                          int ldc,
                                                          int C_mult,
                                                          int C_shift) CUBLAS_NOT_IMPLEMENTED;

/* SYRK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;
/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkEx(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const cuComplex* beta, /* host or device pointer */
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc) CUBLAS_NOT_IMPLEMENTED;

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk3mEx(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha,
                                                      const void* A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const cuComplex* beta,
                                                      void* C,
                                                      cudaDataType Ctype,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;

/* HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const float* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const double* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkEx(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const float* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const float* beta, /* host or device pointer */
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc) CUBLAS_NOT_IMPLEMENTED;

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk3mEx(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float* alpha,
                                                      const void* A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const float* beta,
                                                      void* C,
                                                      cudaDataType Ctype,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;

/* SYR2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float* alpha, /* host or device pointer */
                                                      const float* A,
                                                      int lda,
                                                      const float* B,
                                                      int ldb,
                                                      const float* beta, /* host or device pointer */
                                                      float* C,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double* alpha, /* host or device pointer */
                                                      const double* A,
                                                      int lda,
                                                      const double* B,
                                                      int ldb,
                                                      const double* beta, /* host or device pointer */
                                                      double* C,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha, /* host or device pointer */
                                                      const cuComplex* A,
                                                      int lda,
                                                      const cuComplex* B,
                                                      int ldb,
                                                      const cuComplex* beta, /* host or device pointer */
                                                      cuComplex* C,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex* alpha, /* host or device pointer */
                                                      const cuDoubleComplex* A,
                                                      int lda,
                                                      const cuDoubleComplex* B,
                                                      int ldb,
                                                      const cuDoubleComplex* beta, /* host or device pointer */
                                                      cuDoubleComplex* C,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;
/* HER2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha, /* host or device pointer */
                                                      const cuComplex* A,
                                                      int lda,
                                                      const cuComplex* B,
                                                      int ldb,
                                                      const float* beta, /* host or device pointer */
                                                      cuComplex* C,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex* alpha, /* host or device pointer */
                                                      const cuDoubleComplex* A,
                                                      int lda,
                                                      const cuDoubleComplex* B,
                                                      int ldb,
                                                      const double* beta, /* host or device pointer */
                                                      cuDoubleComplex* C,
                                                      int ldc) CUBLAS_NOT_IMPLEMENTED;
/* SYRKX : eXtended SYRK*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const float* alpha, /* host or device pointer */
                                                   const float* A,
                                                   int lda,
                                                   const float* B,
                                                   int ldb,
                                                   const float* beta, /* host or device pointer */
                                                   float* C,
                                                   int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const double* alpha, /* host or device pointer */
                                                   const double* A,
                                                   int lda,
                                                   const double* B,
                                                   int ldb,
                                                   const double* beta, /* host or device pointer */
                                                   double* C,
                                                   int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuComplex* alpha, /* host or device pointer */
                                                   const cuComplex* A,
                                                   int lda,
                                                   const cuComplex* B,
                                                   int ldb,
                                                   const cuComplex* beta, /* host or device pointer */
                                                   cuComplex* C,
                                                   int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuDoubleComplex* alpha, /* host or device pointer */
                                                   const cuDoubleComplex* A,
                                                   int lda,
                                                   const cuDoubleComplex* B,
                                                   int ldb,
                                                   const cuDoubleComplex* beta, /* host or device pointer */
                                                   cuDoubleComplex* C,
                                                   int ldc) CUBLAS_NOT_IMPLEMENTED;
/* HERKX : eXtended HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuComplex* alpha, /* host or device pointer */
                                                   const cuComplex* A,
                                                   int lda,
                                                   const cuComplex* B,
                                                   int ldb,
                                                   const float* beta, /* host or device pointer */
                                                   cuComplex* C,
                                                   int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuDoubleComplex* alpha, /* host or device pointer */
                                                   const cuDoubleComplex* A,
                                                   int lda,
                                                   const cuDoubleComplex* B,
                                                   int ldb,
                                                   const double* beta, /* host or device pointer */
                                                   cuDoubleComplex* C,
                                                   int ldc) CUBLAS_NOT_IMPLEMENTED;
/* SYMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

/* HEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

/* TRSM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     float* B,
                                                     int ldb) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     double* B,
                                                     int ldb) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* B,
                                                     int ldb) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* B,
                                                     int ldb) CUBLAS_NOT_IMPLEMENTED;

/* TRMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     float* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     double* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     cuComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     cuDoubleComplex* C,
                                                     int ldc) CUBLAS_NOT_IMPLEMENTED;
/* BATCH GEMM */
#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const __half* alpha, /* host or device pointer */
                                                         const __half* const Aarray[],
                                                         int lda,
                                                         const __half* const Barray[],
                                                         int ldb,
                                                         const __half* beta, /* host or device pointer */
                                                         __half* const Carray[],
                                                         int ldc,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;
#endif
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const float* alpha, /* host or device pointer */
                                                         const float* const Aarray[],
                                                         int lda,
                                                         const float* const Barray[],
                                                         int ldb,
                                                         const float* beta, /* host or device pointer */
                                                         float* const Carray[],
                                                         int ldc,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const double* alpha, /* host or device pointer */
                                                         const double* const Aarray[],
                                                         int lda,
                                                         const double* const Barray[],
                                                         int ldb,
                                                         const double* beta, /* host or device pointer */
                                                         double* const Carray[],
                                                         int ldc,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const cuComplex* alpha, /* host or device pointer */
                                                         const cuComplex* const Aarray[],
                                                         int lda,
                                                         const cuComplex* const Barray[],
                                                         int ldb,
                                                         const cuComplex* beta, /* host or device pointer */
                                                         cuComplex* const Carray[],
                                                         int ldc,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mBatched(cublasHandle_t handle,
                                                           cublasOperation_t transa,
                                                           cublasOperation_t transb,
                                                           int m,
                                                           int n,
                                                           int k,
                                                           const cuComplex* alpha, /* host or device pointer */
                                                           const cuComplex* const Aarray[],
                                                           int lda,
                                                           const cuComplex* const Barray[],
                                                           int ldb,
                                                           const cuComplex* beta, /* host or device pointer */
                                                           cuComplex* const Carray[],
                                                           int ldc,
                                                           int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const cuDoubleComplex* alpha, /* host or device pointer */
                                                         const cuDoubleComplex* const Aarray[],
                                                         int lda,
                                                         const cuDoubleComplex* const Barray[],
                                                         int ldb,
                                                         const cuDoubleComplex* beta, /* host or device pointer */
                                                         cuDoubleComplex* const Carray[],
                                                         int ldc,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const void* alpha, /* host or device pointer */
                                                          const void* const Aarray[],
                                                          cudaDataType Atype,
                                                          int lda,
                                                          const void* const Barray[],
                                                          cudaDataType Btype,
                                                          int ldb,
                                                          const void* beta, /* host or device pointer */
                                                          void* const Carray[],
                                                          cudaDataType Ctype,
                                                          int ldc,
                                                          int batchCount,
                                                          cublasComputeType_t computeType,
                                                          cublasGemmAlgo_t algo) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const void* alpha, /* host or device pointer */
                                                                 const void* A,
                                                                 cudaDataType Atype,
                                                                 int lda,
                                                                 long long int strideA, /* purposely signed */
                                                                 const void* B,
                                                                 cudaDataType Btype,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const void* beta, /* host or device pointer */
                                                                 void* C,
                                                                 cudaDataType Ctype,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount,
                                                                 cublasComputeType_t computeType,
                                                                 cublasGemmAlgo_t algo) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const float* alpha, /* host or device pointer */
                                                                const float* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const float* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const float* beta, /* host or device pointer */
                                                                float* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const double* alpha, /* host or device pointer */
                                                                const double* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const double* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const double* beta, /* host or device pointer */
                                                                double* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const cuComplex* alpha, /* host or device pointer */
                                                                const cuComplex* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const cuComplex* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const cuComplex* beta, /* host or device pointer */
                                                                cuComplex* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mStridedBatched(cublasHandle_t handle,
                                                                  cublasOperation_t transa,
                                                                  cublasOperation_t transb,
                                                                  int m,
                                                                  int n,
                                                                  int k,
                                                                  const cuComplex* alpha, /* host or device pointer */
                                                                  const cuComplex* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const cuComplex* B,
                                                                  int ldb,
                                                                  long long int strideB,
                                                                  const cuComplex* beta, /* host or device pointer */
                                                                  cuComplex* C,
                                                                  int ldc,
                                                                  long long int strideC,
                                                                  int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZgemmStridedBatched(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const cuDoubleComplex* alpha, /* host or device pointer */
                          const cuDoubleComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuDoubleComplex* B,
                          int ldb,
                          long long int strideB,
                          const cuDoubleComplex* beta, /* host or device poi */
                          cuDoubleComplex* C,
                          int ldc,
                          long long int strideC,
                          int batchCount) CUBLAS_NOT_IMPLEMENTED;

#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const __half* alpha, /* host or device pointer */
                                                                const __half* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const __half* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const __half* beta, /* host or device pointer */
                                                                __half* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount) CUBLAS_NOT_IMPLEMENTED;
#endif
/* ---------------- CUBLAS BLAS-like extension ---------------- */
/* GEAM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const float* alpha, /* host or device pointer */
                                                  const float* A,
                                                  int lda,
                                                  const float* beta, /* host or device pointer */
                                                  const float* B,
                                                  int ldb,
                                                  float* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const double* alpha, /* host or device pointer */
                                                  const double* A,
                                                  int lda,
                                                  const double* beta, /* host or device pointer */
                                                  const double* B,
                                                  int ldb,
                                                  double* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuComplex* alpha, /* host or device pointer */
                                                  const cuComplex* A,
                                                  int lda,
                                                  const cuComplex* beta, /* host or device pointer */
                                                  const cuComplex* B,
                                                  int ldb,
                                                  cuComplex* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex* alpha, /* host or device pointer */
                                                  const cuDoubleComplex* A,
                                                  int lda,
                                                  const cuDoubleComplex* beta, /* host or device pointer */
                                                  const cuDoubleComplex* B,
                                                  int ldb,
                                                  cuDoubleComplex* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

/* Batched LU - GETRF*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          float* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          double* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          cuComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          cuDoubleComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

/* Batched inversion based on LU factorization from getrf */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const float* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,     /*Device pointer*/
                                                          float* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const double* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,      /*Device pointer*/
                                                          double* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,         /*Device pointer*/
                                                          cuComplex* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuDoubleComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,               /*Device pointer*/
                                                          cuDoubleComplex* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

/* Batched solver based on LU factorization from getrf */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const float* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          float* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const double* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          double* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const cuComplex* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          cuComplex* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const cuDoubleComplex* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          cuDoubleComplex* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

/* TRSM - Batched Triangular Solver */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const float* alpha, /*Host or Device Pointer*/
                                                         const float* const A[],
                                                         int lda,
                                                         float* const B[],
                                                         int ldb,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const double* alpha, /*Host or Device Pointer*/
                                                         const double* const A[],
                                                         int lda,
                                                         double* const B[],
                                                         int ldb,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const cuComplex* alpha, /*Host or Device Pointer*/
                                                         const cuComplex* const A[],
                                                         int lda,
                                                         cuComplex* const B[],
                                                         int ldb,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const cuDoubleComplex* alpha, /*Host or Device Pointer*/
                                                         const cuDoubleComplex* const A[],
                                                         int lda,
                                                         cuDoubleComplex* const B[],
                                                         int ldb,
                                                         int batchCount) CUBLAS_NOT_IMPLEMENTED;

/* Batched - MATINV*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const float* const A[], /*Device pointer*/
                                                           int lda,
                                                           float* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const double* const A[], /*Device pointer*/
                                                           int lda,
                                                           double* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const cuComplex* const A[], /*Device pointer*/
                                                           int lda,
                                                           cuComplex* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const cuDoubleComplex* const A[], /*Device pointer*/
                                                           int lda,
                                                           cuDoubleComplex* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize) CUBLAS_NOT_IMPLEMENTED;

/* Batch QR Factorization */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          float* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          float* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          double* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          double* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          cuComplex* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          cuComplex* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          cuDoubleComplex* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          cuDoubleComplex* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize) CUBLAS_NOT_IMPLEMENTED;
/* Least Square Min only m >= n and Non-transpose supported */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         float* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         float* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray, /*Device pointer*/
                                                         int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         double* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         double* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray, /*Device pointer*/
                                                         int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         cuComplex* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         cuComplex* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray,
                                                         int batchSize) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         cuDoubleComplex* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         cuDoubleComplex* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray,
                                                         int batchSize) CUBLAS_NOT_IMPLEMENTED;
/* DGMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const float* A,
                                                  int lda,
                                                  const float* x,
                                                  int incx,
                                                  float* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const double* A,
                                                  int lda,
                                                  const double* x,
                                                  int incx,
                                                  double* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuComplex* A,
                                                  int lda,
                                                  const cuComplex* x,
                                                  int incx,
                                                  cuComplex* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex* A,
                                                  int lda,
                                                  const cuDoubleComplex* x,
                                                  int incx,
                                                  cuDoubleComplex* C,
                                                  int ldc) CUBLAS_NOT_IMPLEMENTED;

/* TPTTR : Triangular Pack format to Triangular format */
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* AP, float* A, int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* AP, double* A, int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* AP, cuComplex* A, int lda) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpttr(
    cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda) CUBLAS_NOT_IMPLEMENTED;
/* TRTTP : Triangular format to Triangular Pack format */
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, float* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, double* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, cuComplex* AP) CUBLAS_NOT_IMPLEMENTED;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrttp(
    cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP) CUBLAS_NOT_IMPLEMENTED;

#if defined(__cplusplus)
}
#endif /* __cplusplus */
