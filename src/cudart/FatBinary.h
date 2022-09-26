
#pragma once

#include <cuda.h>
#include <unordered_map>
#include <string>

class FatBinary {
	private:
		std::string name;
		void* _cubin;
		void* _ptx;
		std::unordered_map<std::string, struct CUfunc_st *> functions;
	public:
		FatBinary(void* entry);
		inline void* cubin() { return _cubin; }
		int parse();
		struct CUfunc_st* malloc_func_if_necessary(const char *name);
};