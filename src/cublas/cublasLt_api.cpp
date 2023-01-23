
#include <cublasLt.h>
#include <cublas.h>
#include "cublas_header.h"

#define CUBLALTAPI __attribute__((weak)) cublasStatus_t

#include "driver_types.h"
#include "cuComplex.h" /* import complex data type */

#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle);

cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle);

const char* CUBLASWINAPI cublasLtGetStatusName(cublasStatus_t status);

const char* CUBLASWINAPI cublasLtGetStatusString(cublasStatus_t status);

size_t CUBLASWINAPI cublasLtGetVersion(void);

size_t CUBLASWINAPI cublasLtGetCudartVersion(void);

cublasStatus_t CUBLASWINAPI cublasLtGetProperty(libraryPropertyType type, int* value);

/** Execute matrix multiplication (D = alpha * op(A) * op(B) + beta * C).
 *
 * \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
 * \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
 *                                             when workspaceSizeInBytes is less than workspace required by configured
 *                                             algo
 * \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
 *                                             operation
 * \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
 * \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
 * \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmul(cublasLtHandle_t lightHandle,
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
                                           cudaStream_t stream);

/** Matrix layout conversion helper (C = alpha * op(A) + beta * op(B))
 *
 * Can be used to change memory order of data or to scale and shift the values.
 *
 * \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
 * \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
 *                                             when A is not NULL, but Adesc is NULL
 * \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
 *                                             operation
 * \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
 * \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
 * \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixTransform(cublasLtHandle_t lightHandle,
                                                    cublasLtMatrixTransformDesc_t transformDesc,
                                                    const void* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cublasLtMatrixLayout_t Adesc,
                                                    const void* beta, /* host or device pointer */
                                                    const void* B,
                                                    cublasLtMatrixLayout_t Bdesc,
                                                    void* C,
                                                    cublasLtMatrixLayout_t Cdesc,
                                                    cudaStream_t stream);

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatrixLayout_t */
/* ---------------------------------------------------------------------------------------*/

/** Internal. Do not use directly.
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutInit_internal(  //
    cublasLtMatrixLayout_t matLayout,
    size_t size,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld);

/** Initialize matrix layout descriptor in pre-allocated space.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if size of the pre-allocated space is insufficient
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
static inline cublasStatus_t cublasLtMatrixLayoutInit(
    cublasLtMatrixLayout_t matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld) {
  return cublasLtMatrixLayoutInit_internal(matLayout, sizeof(*matLayout), type, rows, cols, ld);
}

/** Create new matrix layout descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(  //
    cublasLtMatrixLayout_t* matLayout,
    cudaDataType type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld);

/** Destroy matrix layout descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);

/** Set matrix layout descriptor attribute.
 *
 * \param[in]  matLayout    The descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutSetAttribute(  //
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    const void* buf,
    size_t sizeInBytes);

/** Get matrix layout descriptor attribute.
 *
 * \param[in]  matLayout    The descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
 *                          bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
 *                                          and buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutGetAttribute(  //
    cublasLtMatrixLayout_t matLayout,
    cublasLtMatrixLayoutAttribute_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten);

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatmulDesc_t */
/* ---------------------------------------------------------------------------------------*/

/** Internal. Do not use directly.
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulDescInit_internal(  //
    cublasLtMatmulDesc_t matmulDesc,
    size_t size,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType);

/** Initialize matmul operation descriptor in pre-allocated space.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if size of the pre-allocated space is insufficient
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was initialized successfully
 */
static inline cublasStatus_t cublasLtMatmulDescInit(  //
    cublasLtMatmulDesc_t matmulDesc,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType) {
  return cublasLtMatmulDescInit_internal(matmulDesc, sizeof(*matmulDesc), computeType, scaleType);
}

/** Create new matmul operation descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType);

/** Destroy matmul operation descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);

/** Set matmul operation descriptor attribute.
 *
 * \param[in]  matmulDesc   The descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulDescSetAttribute(  //
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes);

/** Get matmul operation descriptor attribute.
 *
 * \param[in]  matmulDesc   The descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
 *                          bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
 *                                          and buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulDescGetAttribute(  //
    cublasLtMatmulDesc_t matmulDesc,
    cublasLtMatmulDescAttributes_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten);

/* ---------------------------------------------------------------------------------------*/
/* Helper functions for cublasLtMatrixTransformDesc_t */
/* ---------------------------------------------------------------------------------------*/

/** Matrix transform descriptor attributes to define details of the operation.
 */

/** Internal. Do not use directly.
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescInit_internal(cublasLtMatrixTransformDesc_t transformDesc,
                                                                     size_t size,
                                                                     cudaDataType scaleType);

/** Initialize matrix transform operation descriptor in pre-allocated space.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if size of the pre-allocated space is insufficient
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
static inline cublasStatus_t cublasLtMatrixTransformDescInit(cublasLtMatrixTransformDesc_t transformDesc,
                                                             cudaDataType scaleType) {
  return cublasLtMatrixTransformDescInit_internal(transformDesc, sizeof(*transformDesc), scaleType);
}

/** Create new matrix transform operation descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t* transformDesc,
                                                              cudaDataType scaleType);

/** Destroy matrix transform operation descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc);

/** Set matrix transform operation descriptor attribute.
 *
 * \param[in]  transformDesc  The descriptor
 * \param[in]  attr           The attribute
 * \param[in]  buf            memory address containing the new value
 * \param[in]  sizeInBytes    size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescSetAttribute(  //
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr,
    const void* buf,
    size_t sizeInBytes);

/** Get matrix transform operation descriptor attribute.
 *
 * \param[in]  transformDesc  The descriptor
 * \param[in]  attr           The attribute
 * \param[out] buf            memory address containing the new value
 * \param[in]  sizeInBytes    size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten    only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number
 * of bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
 *                                          and buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
 */
cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescGetAttribute(  //
    cublasLtMatrixTransformDesc_t transformDesc,
    cublasLtMatrixTransformDescAttributes_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten);

/** For computation with complex numbers, this enum allows to apply the Gauss Complexity reduction algorithm
 */

/** Internal. Do not use directly.
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t pref, size_t size);

/** Initialize matmul heuristic search preference descriptor in pre-allocated space.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if size of the pre-allocated space is insufficient
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
static inline cublasStatus_t cublasLtMatmulPreferenceInit(cublasLtMatmulPreference_t pref) {
  return cublasLtMatmulPreferenceInit_internal(pref, sizeof(*pref));
}

/** Create new matmul heuristic search preference descriptor.
 *
 * \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
 * \retval     CUBLAS_STATUS_SUCCESS       if desciptor was created successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref);

/** Destroy matmul heuristic search preference descriptor.
 *
 * \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);

/** Set matmul heuristic search preference descriptor attribute.
 *
 * \param[in]  pref         The descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceSetAttribute(  //
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    const void* buf,
    size_t sizeInBytes);

/** Get matmul heuristic search preference descriptor attribute.
 *
 * \param[in]  pref         The descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
 *                          bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
 *                                          and buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceGetAttribute(  //
    cublasLtMatmulPreference_t pref,
    cublasLtMatmulPreferenceAttributes_t attr,
    void* buf,
    size_t sizeInBytes,
    size_t* sizeWritten);

/** Results structure used by cublasLtMatmulGetAlgo.
 *
 * Holds returned configured algo descriptor and its runtime properties.
 */
typedef struct {
  /** Matmul algorithm descriptor.
   *
   * Must be initialized with cublasLtMatmulAlgoInit() if preferences' CUBLASLT_MATMUL_PERF_SEARCH_MODE is set to
   * CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID
   */
  cublasLtMatmulAlgo_t algo;

  /** Actual size of workspace memory required.
   */
  size_t workspaceSize;

  /** Result status, other fields are only valid if after call to cublasLtMatmulAlgoGetHeuristic() this member is set to
   * CUBLAS_STATUS_SUCCESS.
   */
  cublasStatus_t state;

  /** Waves count - a device utilization metric.
   *
   * wavesCount value of 1.0f suggests that when kernel is launched it will fully occupy the GPU.
   */
  float wavesCount;

  int reserved[4];
} cublasLtMatmulHeuristicResult_t;

/** Query cublasLt heuristic for algorithm appropriate for given use case.
 *
 * \param[in]      lightHandle            Pointer to the allocated cuBLASLt handle for the cuBLASLt
 *                                        context. See cublasLtHandle_t.
 * \param[in]      operationDesc          Handle to the matrix multiplication descriptor.
 * \param[in]      Adesc                  Handle to the layout descriptors for matrix A.
 * \param[in]      Bdesc                  Handle to the layout descriptors for matrix B.
 * \param[in]      Cdesc                  Handle to the layout descriptors for matrix C.
 * \param[in]      Ddesc                  Handle to the layout descriptors for matrix D.
 * \param[in]      preference             Pointer to the structure holding the heuristic search
 *                                        preferences descriptor. See cublasLtMatrixLayout_t.
 * \param[in]      requestedAlgoCount     Size of heuristicResultsArray (in elements) and requested
 *                                        maximum number of algorithms to return.
 * \param[in, out] heuristicResultsArray  Output algorithms and associated runtime characteristics,
 *                                        ordered in increasing estimated compute time.
 * \param[out]     returnAlgoCount        The number of heuristicResultsArray elements written.
 *
 * \retval  CUBLAS_STATUS_INVALID_VALUE   if requestedAlgoCount is less or equal to zero
 * \retval  CUBLAS_STATUS_NOT_SUPPORTED   if no heuristic function available for current configuration
 * \retval  CUBLAS_STATUS_SUCCESS         if query was successful, inspect
 *                                        heuristicResultsArray[0 to (returnAlgoCount - 1)].state
 *                                        for detail status of results
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                                           cublasLtMatmulDesc_t operationDesc,
                                                           cublasLtMatrixLayout_t Adesc,
                                                           cublasLtMatrixLayout_t Bdesc,
                                                           cublasLtMatrixLayout_t Cdesc,
                                                           cublasLtMatrixLayout_t Ddesc,
                                                           cublasLtMatmulPreference_t preference,
                                                           int requestedAlgoCount,
                                                           cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                           int* returnAlgoCount);

/* ---------------------------------------------------------------------------------------*/
/* Lower level API to be able to implement own Heuristic and Find routines                */
/* ---------------------------------------------------------------------------------------*/

/** Routine to get all algo IDs that can potentially run
 *
 * \param[in]  int              requestedAlgoCount requested number of algos (must be less or equal to size of algoIdsA
 * (in elements)) \param[out] algoIdsA         array to write algoIds to \param[out] returnAlgoCount  number of algoIds
 * actually written
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if requestedAlgoCount is less or equal to zero
 * \retval     CUBLAS_STATUS_SUCCESS        if query was successful, inspect returnAlgoCount to get actual number of IDs
 *                                          available
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType,
                                                     cudaDataType_t Atype,
                                                     cudaDataType_t Btype,
                                                     cudaDataType_t Ctype,
                                                     cudaDataType_t Dtype,
                                                     int requestedAlgoCount,
                                                     int algoIdsArray[],
                                                     int* returnAlgoCount);

/** Initialize algo structure
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if algo is NULL or algoId is outside of recognized range
 * \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algoId is not supported for given combination of data types
 * \retval     CUBLAS_STATUS_SUCCESS        if the structure was successfully initialized
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle,
                                                   cublasComputeType_t computeType,
                                                   cudaDataType_t scaleType,
                                                   cudaDataType_t Atype,
                                                   cudaDataType_t Btype,
                                                   cudaDataType_t Ctype,
                                                   cudaDataType_t Dtype,
                                                   int algoId,
                                                   cublasLtMatmulAlgo_t* algo);

/** Check configured algo descriptor for correctness and support on current device.
 *
 * Result includes required workspace size and calculated wave count.
 *
 * CUBLAS_STATUS_SUCCESS doesn't fully guarantee algo will run (will fail if e.g. buffers are not correctly aligned);
 * but if cublasLtMatmulAlgoCheck fails, the algo will not run.
 *
 * \param[in]  algo    algo configuration to check
 * \param[out] result  result structure to report algo runtime characteristics; algo field is never updated
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if matrix layout descriptors or operation descriptor don't match algo
 *                                          descriptor
 * \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algo configuration or data type combination is not currently supported on
 *                                          given device
 * \retval     CUBLAS_STATUS_ARCH_MISMATCH  if algo configuration cannot be run using the selected device
 * \retval     CUBLAS_STATUS_SUCCESS        if check was successful
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCheck(  //
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,  ///< may point to result->algo
    cublasLtMatmulHeuristicResult_t* result);

/** Capabilities Attributes that can be retrieved from an initialized Algo structure
 */
typedef enum {
  /** support for split K, see CUBLASLT_ALGO_CONFIG_SPLITK_NUM
   *
   * int32_t, 0 means no support, supported otherwise
   */
  CUBLASLT_ALGO_CAP_SPLITK_SUPPORT = 0,
  /** reduction scheme mask, see cublasLtReductionScheme_t; shows supported reduction schemes, if reduction scheme is
   * not masked out it is supported.
   *
   * e.g. int isReductionSchemeComputeTypeSupported ? (reductionSchemeMask & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) ==
   * CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE ? 1 : 0;
   *
   * uint32_t
   */
  CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK = 1,
  /** support for cta swizzling, see CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
   *
   * uint32_t, 0 means no support, 1 means supported value of 1, other values are reserved
   */
  CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT = 2,
  /** support strided batch
   *
   * int32_t, 0 means no support, supported otherwise
   */
  CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT = 3,
  /** support results out of place (D != C in D = alpha.A.B + beta.C)
   *
   * int32_t, 0 means no support, supported otherwise
   */
  CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT = 4,
  /** syrk/herk support (on top of regular gemm)
   *
   * int32_t, 0 means no support, supported otherwise
   */
  CUBLASLT_ALGO_CAP_UPLO_SUPPORT = 5,
  /** tile ids possible to use, see cublasLtMatmulTile_t; if no tile ids are supported use
   * CUBLASLT_MATMUL_TILE_UNDEFINED
   *
   * use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count
   *
   * array of uint32_t
   */
  CUBLASLT_ALGO_CAP_TILE_IDS = 6,
  /** custom option range is from 0 to CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX (inclusive), see
   * CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
   *
   * int32_t
   */
  CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX = 7,
  /** whether algorithm is using regular compute or tensor operations
   *
   * int32_t 0 means regular compute, 1 means tensor operations;
   * DEPRECATED
   */
  CUBLASLT_ALGO_CAP_MATHMODE_IMPL = 8,
  /** whether algorithm implements gaussian optimization of complex matrix multiplication, see cublasMath_t
   *
   * int32_t 0 means regular compute, 1 means gaussian;
   * DEPRECATED
   */
  CUBLASLT_ALGO_CAP_GAUSSIAN_IMPL = 9,
  /** whether algorithm supports custom (not COL or ROW memory order), see cublasLtOrder_t
   *
   * int32_t 0 means only COL and ROW memory order is allowed, non-zero means that algo might have different
   * requirements;
   */
  CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER = 10,

  /** bitmask enumerating pointer modes algorithm supports
   *
   * uint32_t, see cublasLtPointerModeMask_t
   */
  CUBLASLT_ALGO_CAP_POINTER_MODE_MASK = 11,

  /** bitmask enumerating kinds of postprocessing algorithm supports in the epilogue
   *
   * uint32_t, see cublasLtEpilogue_t
   */
  CUBLASLT_ALGO_CAP_EPILOGUE_MASK = 12,
  /** stages ids possible to use, see cublasLtMatmulStages_t; if no stages ids are supported use
   * CUBLASLT_MATMUL_STAGES_UNDEFINED
   *
   * use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count
   *
   * array of uint32_t
   */
  CUBLASLT_ALGO_CAP_STAGES_IDS = 13,
  /** support for nagative ld for all of the matrices
   *
   * int32_t 0 means no support, supported otherwise
   */
  CUBLASLT_ALGO_CAP_LD_NEGATIVE = 14,
  /** details about algorithm's implementation that affect it's numerical behavior
   *
   * uint64_t, see cublasLtNumericalImplFlags_t
   */
  CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS = 15,
  /** minimum alignment required for A matrix in bytes
   *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
   *
   * uint32_t
   */
  CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES = 16,
  /** minimum alignment required for B matrix in bytes
   *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
   *
   * uint32_t
   */
  CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES = 17,
  /** minimum alignment required for C matrix in bytes
   *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
   *
   * uint32_t
   */
  CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES = 18,
  /** minimum alignment required for D matrix in bytes
   *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
   *
   * uint32_t
   */
  CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES = 19,
} cublasLtMatmulAlgoCapAttributes_t;

/** Get algo capability attribute.
 *
 * E.g. to get list of supported Tile IDs:
 *      cublasLtMatmulTile_t tiles[CUBLASLT_MATMUL_TILE_END];
 *      size_t num_tiles, size_written;
 *      if (cublasLtMatmulAlgoCapGetAttribute(algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof(tiles), size_written) ==
 * CUBLAS_STATUS_SUCCESS) { num_tiles = size_written / sizeof(tiles[0]);
 *      }
 *
 * \param[in]  algo         The algo descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
 *                          bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
 *                                          and buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t* algo,
                                                              cublasLtMatmulAlgoCapAttributes_t attr,
                                                              void* buf,
                                                              size_t sizeInBytes,
                                                              size_t* sizeWritten);

/** Algo Configuration Attributes that can be set according to the Algo capabilities
 */
typedef enum {
  /** algorithm index, see cublasLtMatmulAlgoGetIds()
   *
   * readonly, set by cublasLtMatmulAlgoInit()
   * int32_t
   */
  CUBLASLT_ALGO_CONFIG_ID = 0,
  /** tile id, see cublasLtMatmulTile_t
   *
   * uint32_t, default: CUBLASLT_MATMUL_TILE_UNDEFINED
   */
  CUBLASLT_ALGO_CONFIG_TILE_ID = 1,
  /** number of K splits, if != 1, SPLITK_NUM parts of matrix multiplication will be computed in parallel,
   * and then results accumulated according to REDUCTION_SCHEME
   *
   * uint32_t, default: 1
   */
  CUBLASLT_ALGO_CONFIG_SPLITK_NUM = 2,
  /** reduction scheme, see cublasLtReductionScheme_t
   *
   * uint32_t, default: CUBLASLT_REDUCTION_SCHEME_NONE
   */
  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME = 3,
  /** cta swizzling, change mapping from CUDA grid coordinates to parts of the matrices
   *
   * possible values: 0, 1, other values reserved
   *
   * uint32_t, default: 0
   */
  CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING = 4,
  /** custom option, each algorithm can support some custom options that don't fit description of the other config
   * attributes, see CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX to get accepted range for any specific case
   *
   * uint32_t, default: 0
   */
  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION = 5,
  /** stages id, see cublasLtMatmulStages_t
   *
   * uint32_t, default: CUBLASLT_MATMUL_STAGES_UNDEFINED
   */
  CUBLASLT_ALGO_CONFIG_STAGES_ID = 6,
} cublasLtMatmulAlgoConfigAttributes_t;

/** Set algo configuration attribute.
 *
 * \param[in]  algo         The algo descriptor
 * \param[in]  attr         The attribute
 * \param[in]  buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t* algo,
                                                                 cublasLtMatmulAlgoConfigAttributes_t attr,
                                                                 const void* buf,
                                                                 size_t sizeInBytes);

/** Get algo configuration attribute.
 *
 * \param[in]  algo         The algo descriptor
 * \param[in]  attr         The attribute
 * \param[out] buf          memory address containing the new value
 * \param[in]  sizeInBytes  size of buf buffer for verification (in bytes)
 * \param[out] sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
 *                          bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
 *                                          and buf is NULL or sizeInBytes doesn't match size of internal storage for
 *                                          selected attribute
 * \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
 */
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t* algo,
                                                                 cublasLtMatmulAlgoConfigAttributes_t attr,
                                                                 void* buf,
                                                                 size_t sizeInBytes,
                                                                 size_t* sizeWritten);

/** Experimental: Logger callback type.
 */
typedef void (*cublasLtLoggerCallback_t)(int logLevel, const char* functionName, const char* message);

/** Experimental: Logger callback setter.
 *
 * \param[in]  callback                     a user defined callback function to be called by the logger
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if callback was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback);

/** Experimental: Log file setter.
 *
 * \param[in]  file                         an open file with write permissions
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log file was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtLoggerSetFile(FILE* file);

/** Experimental: Open log file.
 *
 * \param[in]  logFile                      log file path. if the log file does not exist, it will be created
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log file was created successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtLoggerOpenFile(const char* logFile);

/** Experimental: Log level setter.
 *
 * \param[in]  level                        log level, should be one of the following:
 *                                          0. Off
 *                                          1. Errors
 *                                          2. Performance Trace
 *                                          3. Performance Hints
 *                                          4. Heuristics Trace
 *                                          5. API Trace
 *
 * \retval     CUBLAS_STATUS_INVALID_VALUE  if log level is not one of the above levels
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log level was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtLoggerSetLevel(int level);

/** Experimental: Log mask setter.
 *
 * \param[in]  mask                         log mask, should be a combination of the following masks:
 *                                          0.  Off
 *                                          1.  Errors
 *                                          2.  Performance Trace
 *                                          4.  Performance Hints
 *                                          8.  Heuristics Trace
 *                                          16. API Trace
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if log mask was set successfully
 */
cublasStatus_t CUBLASWINAPI cublasLtLoggerSetMask(int mask);

/** Experimental: Disable logging for the entire session.
 *
 * \retval     CUBLAS_STATUS_SUCCESS        if disabled logging
 */
cublasStatus_t CUBLASWINAPI cublasLtLoggerForceDisable();

#if defined(__cplusplus)
}
#endif /* __cplusplus */
