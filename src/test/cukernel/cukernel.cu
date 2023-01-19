
#include "cukernel.h"

#define N 100

// typedef __device__ void (*lambda_kernel)(int* c, int* a, int* b, int i);

template<class F>
__global__ void add(int* c, int* a, int* b, int size, F kernel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		kernel(c, a, b, i);
	}
}

#include <stdio.h>

extern "C" int main(int argc, char* argv[]) {

	int *darra, *darrb, *darrc;
	int all_correct = 1;
	int n;

	if (argc == 1) {
		n = 100;
	} else {
		n = atoi(argv[1]);
	}

	int *arra = (int*) malloc(sizeof(int) * (n * n));
	int *arrb = (int*) malloc(sizeof(int) * (n * n));
	int *arrc = (int*) malloc(sizeof(int) * (n * n));

	cudaMalloc((void**)&darra, (n * n) * sizeof(int));
	cudaMalloc((void**)&darrb, (n * n) * sizeof(int));
	cudaMalloc((void**)&darrc, (n * n) * sizeof(int));

	for (int i = 0;i < (n * n);i++) {
		arra[i] = i;
		arrb[i] = i;
	}

	cudaMemcpy(darra, arra, (n * n) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(darrb, arrb, (n * n) * sizeof(int), cudaMemcpyHostToDevice);

	int block  = 100;
	int grid = n * n / block;

	auto lam = [=] __device__(int* c, int* a, int* b, int i) {
		c[i] = a[i] + b[i];
	};

	add<<<grid, block>>>(darrc, darra, darrb, n * n, lam);

	cudaDeviceSynchronize();

	cudaMemcpy(arrc, darrc, (n * n)*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0;i < (n * n);i++) {
		if (arrc[i] != arra[i] + arrb[i]) {
			all_correct = 0;
			printf("r[%d] is incorrect\n", i);
		}
	}

	if (all_correct) {
		fprintf(stderr, "all r[i] is correct\n");
	} else {
		fprintf(stderr, "some r[i] are incorrect\n");
	}

	cudaFree(darra);
	cudaFree(darrb);
	cudaFree(darrc);

	return all_correct;
}
