
#include <cublasXt.h>
#include <cublas.h>
#include "cublas_header.h"

#define cublasStatus_t __attribute__((weak)) cublasStatus_t

#include "driver_types.h"
#include "cuComplex.h" /* import complex data type */

#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

cublasStatus_t CUBLASWINAPI cublasXtCreate(cublasXtHandle_t* handle) CUBLAS_NOT_IMPLEMENTED;
cublasStatus_t CUBLASWINAPI cublasXtDestroy(cublasXtHandle_t handle) CUBLAS_NOT_IMPLEMENTED;
cublasStatus_t CUBLASWINAPI cublasXtGetNumBoards(int nbDevices, int deviceId[], int* nbBoards) CUBLAS_NOT_IMPLEMENTED;
cublasStatus_t CUBLASWINAPI cublasXtMaxBoards(int* nbGpuBoards) CUBLAS_NOT_IMPLEMENTED;
/* This routine selects the Gpus that the user want to use for CUBLAS-XT */
cublasStatus_t CUBLASWINAPI cublasXtDeviceSelect(cublasXtHandle_t handle, int nbDevices, int deviceId[]) CUBLAS_NOT_IMPLEMENTED;

/* This routine allows to change the dimension of the tiles ( blockDim x blockDim ) */
cublasStatus_t CUBLASWINAPI cublasXtSetBlockDim(cublasXtHandle_t handle, int blockDim) CUBLAS_NOT_IMPLEMENTED;
cublasStatus_t CUBLASWINAPI cublasXtGetBlockDim(cublasXtHandle_t handle, int* blockDim) CUBLAS_NOT_IMPLEMENTED;

/* This routine allows to CUBLAS-XT to pin the Host memory if it find out that some of the matrix passed
   are not pinned : Pinning/Unpinning the Host memory is still a costly operation
   It is better if the user controls the memory on its own (by pinning/unpinning oly when necessary)
*/
cublasStatus_t CUBLASWINAPI cublasXtGetPinningMemMode(cublasXtHandle_t handle, cublasXtPinnedMemMode_t* mode) CUBLAS_NOT_IMPLEMENTED;
cublasStatus_t CUBLASWINAPI cublasXtSetPinningMemMode(cublasXtHandle_t handle, cublasXtPinnedMemMode_t mode) CUBLAS_NOT_IMPLEMENTED;

/* This routines is to provide a CPU Blas routines, used for too small sizes or hybrid computation */

/* Currently only 32-bit integer BLAS routines are supported */
cublasStatus_t CUBLASWINAPI cublasXtSetCpuRoutine(cublasXtHandle_t handle,
                                                  cublasXtBlasOp_t blasOp,
                                                  cublasXtOpType_t type,
                                                  void* blasFunctor) CUBLAS_NOT_IMPLEMENTED;

/* Specified the percentage of work that should done by the CPU, default is 0 (no work) */
cublasStatus_t CUBLASWINAPI cublasXtSetCpuRatio(cublasXtHandle_t handle,
                                                cublasXtBlasOp_t blasOp,
                                                cublasXtOpType_t type,
                                                float ratio) CUBLAS_NOT_IMPLEMENTED;

/* GEMM */
cublasStatus_t CUBLASWINAPI cublasXtSgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;
/* ------------------------------------------------------- */
/* SYRK */
cublasStatus_t CUBLASWINAPI cublasXtSsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* beta,
                                          float* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* beta,
                                          double* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;
/* -------------------------------------------------------------------- */
/* HERK */
cublasStatus_t CUBLASWINAPI cublasXtCherk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const float* beta,
                                          cuComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZherk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const double* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;
/* -------------------------------------------------------------------- */
/* SYR2K */
cublasStatus_t CUBLASWINAPI cublasXtSsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const float* alpha,
                                           const float* A,
                                           size_t lda,
                                           const float* B,
                                           size_t ldb,
                                           const float* beta,
                                           float* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const double* alpha,
                                           const double* A,
                                           size_t lda,
                                           const double* B,
                                           size_t ldb,
                                           const double* beta,
                                           double* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const cuComplex* beta,
                                           cuComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const cuDoubleComplex* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;
/* -------------------------------------------------------------------- */
/* HERKX : variant extension of HERK */
cublasStatus_t CUBLASWINAPI cublasXtCherkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const float* beta,
                                           cuComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZherkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const double* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

/* -------------------------------------------------------------------- */
/* TRSM */
cublasStatus_t CUBLASWINAPI cublasXtStrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          float* B,
                                          size_t ldb) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDtrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          double* B,
                                          size_t ldb) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCtrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          cuComplex* B,
                                          size_t ldb) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZtrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          cuDoubleComplex* B,
                                          size_t ldb) CUBLAS_NOT_IMPLEMENTED;
/* -------------------------------------------------------------------- */
/* SYMM : Symmetric Multiply Matrix*/
cublasStatus_t CUBLASWINAPI cublasXtSsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;
/* -------------------------------------------------------------------- */
/* HEMM : Hermitian Matrix Multiply */
cublasStatus_t CUBLASWINAPI cublasXtChemm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZhemm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

/* -------------------------------------------------------------------- */
/* SYRKX : variant extension of SYRK  */
cublasStatus_t CUBLASWINAPI cublasXtSsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const float* alpha,
                                           const float* A,
                                           size_t lda,
                                           const float* B,
                                           size_t ldb,
                                           const float* beta,
                                           float* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const double* alpha,
                                           const double* A,
                                           size_t lda,
                                           const double* B,
                                           size_t ldb,
                                           const double* beta,
                                           double* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const cuComplex* beta,
                                           cuComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const cuDoubleComplex* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;
/* -------------------------------------------------------------------- */
/* HER2K : variant extension of HERK  */
cublasStatus_t CUBLASWINAPI cublasXtCher2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const float* beta,
                                           cuComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZher2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const double* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc) CUBLAS_NOT_IMPLEMENTED;

/* -------------------------------------------------------------------- */
/* SPMM : Symmetric Packed Multiply Matrix*/
cublasStatus_t CUBLASWINAPI cublasXtSspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* AP,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* AP,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* AP,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* AP,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

/* -------------------------------------------------------------------- */
/* TRMM */
cublasStatus_t CUBLASWINAPI cublasXtStrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          float* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtDtrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          double* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtCtrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          cuComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

cublasStatus_t CUBLASWINAPI cublasXtZtrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          cuDoubleComplex* C,
                                          size_t ldc) CUBLAS_NOT_IMPLEMENTED;

#if defined(__cplusplus)
}
#endif /* __cplusplus */