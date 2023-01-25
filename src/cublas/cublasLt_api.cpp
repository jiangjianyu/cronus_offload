
#include <cublasLt.h>
#include "cublas_header.h"

#define CUBLALTAPI __attribute__((weak)) CUBLASLTAPI cublasStatus_t

#include "driver_types.h"
#include "cuComplex.h" /* import complex data type */

#include "cublas_v2.h"

#include <cublas_api.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define CUBLASLTAPI __attribute__((weak))

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle) CUBLAS_NOT_IMPLEMENTED;

const char* CUBLASWINAPI cublasLtGetStatusName(cublasStatus_t status) { return cublasGetStatusName(status); }
const char* CUBLASWINAPI cublasLtGetStatusString(cublasStatus_t status) { return cublasGetStatusString(status); }

size_t CUBLASWINAPI cublasLtGetVersion(void) { return CUBLAS_VERSION; }
size_t CUBLASWINAPI cublasLtGetCudartVersion(void) { return CUBLAS_VERSION; }

CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtGetProperty(libraryPropertyType type, int* value) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmul(cublasLtHandle_t lightHandle,
                                           cublasLtMatmulDesc_t computeDesc,
                                           const void* alpha, /* host or device pointer */
                                           const void* A,
                                           cublasLtMatrixLayout_t Adesc,
                                           const void* B,
                                           cublasLtMatrixLayout_t Bdesc,
                                           const void* beta, /* host or device pointer */
                                           const void* C,
                                           cublasLtMatrixLayout_t Cdesc,
                                           void* D,
                                           cublasLtMatrixLayout_t Ddesc,
                                           const cublasLtMatmulAlgo_t* algo,
                                           void* workspace,
                                           size_t workspaceSizeInBytes,
                                           cudaStream_t stream) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixTransform(cublasLtHandle_t lightHandle,
                                                    cublasLtMatrixTransformDesc_t transformDesc,
                                                    const void* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cublasLtMatrixLayout_t Adesc,
                                                    const void* beta, /* host or device pointer */
                                                    const void* B,
                                                    cublasLtMatrixLayout_t Bdesc,
                                                    void* C,
                                                    cublasLtMatrixLayout_t Cdesc,
                                                    cudaStream_t stream) CUBLAS_NOT_IMPLEMENTED;

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatrixLayout_t */
/* ---------------------------------------------------------------------------------------*/

/** Internal. Do not use directly.
 */
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutInit_internal(  //
    cublasLtMatrixLayout_t matLayout,
    size_t size,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld) CUBLAS_NOT_IMPLEMENTED;
/*
static inline CUBLASLTAPI cublasStatus_t cublasLtMatrixLayoutInit(
    cublasLtMatrixLayout_t matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld) {
  return cublasLtMatrixLayoutInit_internal(matLayout, sizeof(*matLayout), type, rows, cols, ld) CUBLAS_NOT_IMPLEMENTED;
}
*/
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(  //
    cublasLtMatrixLayout_t* matLayout,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutSetAttribute(  //
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void* buf,
    size_t sizeInBytes) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutGetAttribute(  //
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten) CUBLAS_NOT_IMPLEMENTED;

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatmulDesc_t */
/* ---------------------------------------------------------------------------------------*/

/** Internal. Do not use directly.
 */
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulDescInit_internal(  //
    cublasLtMatmulDesc_t matmulDesc,
    size_t size,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType) CUBLAS_NOT_IMPLEMENTED;
/*
static inline CUBLASLTAPI cublasStatus_t cublasLtMatmulDescInit(  //
    cublasLtMatmulDesc_t matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType) {
  return cublasLtMatmulDescInit_internal(matmulDesc, sizeof(*matmulDesc), computeType, scaleType) CUBLAS_NOT_IMPLEMENTED;
}
*/
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulDescSetAttribute(  //
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulDescGetAttribute(  //
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten) CUBLAS_NOT_IMPLEMENTED;

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatrixTransformDesc_t */
/* ---------------------------------------------------------------------------------------*/

/** Matrix transform descriptor attributes to define details of the operation.
 */

/** Internal. Do not use directly.
 */
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescInit_internal(cublasLtMatrixTransformDesc_t transformDesc,
                                                                     size_t size,
                                                                     cudaDataType scaleType) CUBLAS_NOT_IMPLEMENTED;
/*
static inline CUBLASLTAPI cublasStatus_t cublasLtMatrixTransformDescInit(cublasLtMatrixTransformDesc_t transformDesc,
                                                             cudaDataType scaleType) {
  return cublasLtMatrixTransformDescInit_internal(transformDesc, sizeof(*transformDesc), scaleType) CUBLAS_NOT_IMPLEMENTED;
}
*/
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t* transformDesc,
                                                              cudaDataType scaleType) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescSetAttribute(  //
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescGetAttribute(  //
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten) CUBLAS_NOT_IMPLEMENTED;

/** For computation with complex numbers, this enum allows to apply the Gauss Complexity reduction algorithm
 */

/** Internal. Do not use directly.
 */
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t pref, size_t size) CUBLAS_NOT_IMPLEMENTED;
/*
static inline CUBLASLTAPI cublasStatus_t cublasLtMatmulPreferenceInit(cublasLtMatmulPreference_t pref) {
  return cublasLtMatmulPreferenceInit_internal(pref, sizeof(*pref)) CUBLAS_NOT_IMPLEMENTED;
}
*/
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceSetAttribute(  //
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    const void* buf,
    size_t sizeInBytes) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceGetAttribute(  //
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                                           cublasLtMatmulDesc_t operationDesc,
                                                           cublasLtMatrixLayout_t Adesc,
                                                           cublasLtMatrixLayout_t Bdesc,
                                                           cublasLtMatrixLayout_t Cdesc,
                                                           cublasLtMatrixLayout_t Ddesc,
                                                           cublasLtMatmulPreference_t preference,
                                                           int requestedAlgoCount,
                                                           cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                           int* returnAlgoCount) CUBLAS_NOT_IMPLEMENTED;

/* ---------------------------------------------------------------------------------------*/
/* Lower level API to be able to implement own Heuristic and Find routines                */
/* ---------------------------------------------------------------------------------------*/
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType,
                                                     cudaDataType_t Atype,
                                                     cudaDataType_t Btype,
                                                     cudaDataType_t Ctype,
                                                     cudaDataType_t Dtype,
                                                     int requestedAlgoCount,
                                                     int algoIdsArray[],
                                                     int* returnAlgoCount) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle,
                                                   cublasComputeType_t computeType,
                                                   cudaDataType_t scaleType,
                                                   cudaDataType_t Atype,
                                                   cudaDataType_t Btype,
                                                   cudaDataType_t Ctype,
                                                   cudaDataType_t Dtype,
                                                   int algoId,
                                                   cublasLtMatmulAlgo_t* algo) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCheck(  //
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,  ///< may point to result->algo
    cublasLtMatmulHeuristicResult_t* result) CUBLAS_NOT_IMPLEMENTED;

/** Capabilities Attributes that can be retrieved from an initialized Algo structure
 */
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t* algo,
                                                              cublasLtMatmulAlgoCapAttributes_t attr,
                                                              void* buf,
                                                              size_t sizeInBytes,
                                                              size_t* sizeWritten) CUBLAS_NOT_IMPLEMENTED;

/** Algo Configuration Attributes that can be set according to the Algo capabilities
 */
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t* algo,
                                                                 cublasLtMatmulAlgoConfigAttributes_t attr,
                                                                 const void* buf,
                                                                 size_t sizeInBytes) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo,
                                                                 cublasLtMatmulAlgoConfigAttributes_t attr,
                                                                 void* buf,
                                                                 size_t sizeInBytes,
                                                                 size_t* sizeWritten) CUBLAS_NOT_IMPLEMENTED;

CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtLoggerSetFile(FILE* file) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtLoggerOpenFile(const char* logFile) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtLoggerSetLevel(int level) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtLoggerSetMask(int mask) CUBLAS_NOT_IMPLEMENTED;
CUBLASLTAPI cublasStatus_t CUBLASWINAPI cublasLtLoggerForceDisable() CUBLAS_NOT_IMPLEMENTED;

#if defined(__cplusplus)
}
#endif /* __cplusplus */