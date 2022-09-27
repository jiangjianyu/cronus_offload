
#pragma once

#include <cuda.h>
#include <unordered_map>
#include <string>
#include <vector>

#define NVIDIA_CONST_SEGMENT_MAX_COUNT 10

#define gdev_cuda_raw_func CUfunc_st

struct cuda_param {
	int idx;
	uint32_t offset;
	uint32_t size;
	uint32_t flags;
};

struct cuda_raw_func {
	const char *name;
	void *code_buf;
	uint32_t code_size;
	struct {
		void *buf;
		uint32_t size;
	} cmem[NVIDIA_CONST_SEGMENT_MAX_COUNT]; /* local to functions. */
	uint32_t reg_count;
	uint32_t bar_count;
	uint32_t stack_depth;
	uint32_t stack_size;
	uint32_t shared_size;
	uint32_t param_base;
	uint32_t param_size;
	uint32_t param_count;
	std::vector<struct cuda_param> param_data;
	uint32_t local_size;
	uint32_t local_size_neg;
};

class FatBinary {
	private:
		std::string name;
		void* _cubin;
		void* _ptx;
		std::unordered_map<std::string, struct cuda_raw_func *> functions;
	public:
		FatBinary(void* entry);
		inline void* cubin() { return _cubin; }
		int parse();
		struct cuda_raw_func* malloc_func_if_necessary(const char *name);
		struct cuda_raw_func* get_function(const char *name);
};