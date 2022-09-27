
#include "cukernel.h"

#define N 100

__global__ void addKernel(int* c, int* a, int* b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}