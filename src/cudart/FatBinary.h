
#pragma once

#include <string>

class FatBinary {
	private:
		std::string name;
		void* _cubin;
		void* _ptx;
	public:
		FatBinary(void* entry);
		inline void* cubin() { return _cubin; }
};