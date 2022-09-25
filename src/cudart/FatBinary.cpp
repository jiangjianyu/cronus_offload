/*!
	\file FatBinaryContext.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief object for interacting with CUDA Fat Binaries
*/

#include "cudaFatBinary.h"

#include <sstream>
#include "FatBinary.h"

#include <iostream>

#define report(s) { 					\
	std::stringstream ss;				\
	ss << s; 							\
	fprintf(stderr, "%s\n", ss.str().c_str());	\
	}

#define assertM(cond, s) { 						\
	std::stringstream ss;						\
	ss << s; 									\
	fprintf(stderr, "%s\n", ss.str().c_str());	\
	}

FatBinary::FatBinary(void* ptr) {
	report("FatBinaryContext(" << ptr << ")");

	char* _name = 0;
	_ptx = 0;
	_cubin = 0;

	void* cubin_ptr = ptr;

	if(*(int*)cubin_ptr == __cudaFatMAGIC) {
		__cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)cubin_ptr;

		_name = binary->ident;

		//assertM(binary->ptx != 0, "binary contains no PTX");
		//assertM(binary->ptx->ptx != 0, "binary contains no PTX");

		unsigned int ptxVersion = 0;
		unsigned int cubinVersion = 0;

		if (binary->ptx) {
			report("Getting the highest PTX version");

			for(unsigned int i = 0; ; ++i)
			{
				if((binary->ptx[i].ptx) == 0) break;
		
				std::string computeCapability = binary->ptx[i].gpuProfileName;
				std::string versionString(computeCapability.begin() + 8,
					computeCapability.end());
		
				std::stringstream version;
				unsigned int thisVersion = 0;
			
				version << versionString;
				version >> thisVersion;
				if(thisVersion > ptxVersion)
				{
					ptxVersion = thisVersion;
					_ptx = binary->ptx[i].ptx;
				}
			}		
			report(" Selected version " << ptxVersion);
		}
		if (binary->cubin) {
			report("Getting the highest CUBIN version");

			for(unsigned int i = 0; ; ++i)
			{
				if((binary->cubin[i].cubin) == 0) break;
		
				std::string computeCapability = binary->cubin[i].gpuProfileName;
				std::string versionString(computeCapability.begin() + 8,
					computeCapability.end());
		
				std::stringstream version;
				unsigned int thisVersion = 0;
			
				version << versionString;
				version >> thisVersion;
				if(thisVersion > cubinVersion)
				{
					cubinVersion = thisVersion;
					_cubin = binary->cubin[i].cubin;
				}
			}		
			report(" Selected version " << cubinVersion);
		}
	}
	else if (*(int*)cubin_ptr == __cudaFatMAGIC2) {
		report("Found new fat binary format!");
		__cudaFatCudaBinary2* binary = (__cudaFatCudaBinary2*) cubin_ptr;
		__cudaFatCudaBinary2Header* header =
			(__cudaFatCudaBinary2Header*) binary->fatbinData;
		
		report(" binary size is: " << header->length << " bytes");
				
		char* base = (char*)(header + 1);
		long long unsigned int offset = 0;
		__cudaFatCudaBinary2EntryRec* entry = (__cudaFatCudaBinary2EntryRec*)(base);
		

		while (offset < header->length) {
			_name = (char*)entry + entry->name;
			if (entry->type & FATBIN_2_PTX) {
				if (!_ptx) {
					_ptx  = (char*)entry + entry->binary;
					if(entry->flags & COMPRESSED_PTX)
					{
						report("compressed ptx\n");
						_cubin = 0;
						_ptx = 0;
						return;
					}
				}
			}
			if (entry->type & FATBIN_2_ELF) {
				if (!_cubin)
					_cubin  = (char*)entry + entry->binary;
			}


			entry = (__cudaFatCudaBinary2EntryRec*)(base + offset);
			offset += entry->binary + entry->binarySize;
		}

	}
	else {
		assertM(false, "unknown fat binary magic number "
			<< std::hex << *(int*)cubin_ptr);		
	}
	
	if (!_ptx) {
		report("registered, contains NO PTX");
	}
	else {
		report("registered, contains PTX");	
	}

	if (!_cubin) {
		report("registered, contains NO CUBIN");
	}
	else {
		report("registered, contains CUBIN");	
	}
}
