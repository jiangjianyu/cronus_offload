
#include "rpc/rpc.h"
#include "dispatch_header.h"
#include "cuda_runtime_t.h"
#include "cuda_driver_t.h"

#define DISPATCH(s) case s##_DISPATCHER: return s##_dispatch(buffer);

int rpc_dispatch(int dispatch_idx, char* buffer) {
    switch (dispatch_idx) {
        DISPATCH(cuda_runtime)
        DISPATCH(cuda_driver)
        default: fprintf(stderr, "no dispatcher found!!!"); return -1;
    }
}