
extern int do_kernel();

#include "cuda_runtime.h"

int main() {
    void *devptr;
    int r = do_kernel();
    cudaMalloc(&devptr, 10);
    return r;
}
