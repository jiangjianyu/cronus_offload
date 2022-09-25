/*
 * Copyright (C) 2011 Shinpei Kato
 *
 * Systems Research Lab, University of California at Santa Cruz
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "cuda.h"
#include "cuda_elf.h"
#include "cuda_version.h"

#include <elf.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/errno.h>

#if 0
#define Elf_Ehdr Elf32_Ehdr
#define Elf_Shdr Elf32_Shdr
#define Elf_Phdr Elf32_Phdr
#define Elf_Sym	 Elf32_Sym
#else
#define Elf_Ehdr Elf64_Ehdr
#define Elf_Shdr Elf64_Shdr
#define Elf_Phdr Elf64_Phdr
#define Elf_Sym	 Elf64_Sym
#endif

#define GDEV_PRINT(s...) fprintf(stderr, s);
#define MALLOC(s) malloc(s)

#define SH_TEXT ".text."
#define SH_INFO ".nv.info"
#define SH_INFO_FUNC ".nv.info."
#define SH_LOCAL ".nv.local."
#define SH_SHARED ".nv.shared."
#define SH_CONST ".nv.constant"
#define SH_REL ".rel.nv.constant"
#define SH_RELSPACE ".nv.constant14"
#define SH_GLOBAL ".nv.global"
#define SH_GLOBAL_INIT ".nv.global.init"
#define NV_GLOBAL   0x10

/* Macros of nvinfo 
 * This is from https://github.com/cloudcores/CuAssembler
 * But it may not be correct
 */
#define EIATTR_CTAIDZ_USED 0x0401
#define EIATTR_MAX_THREADS 0x0504
#define EIATTR_PARAM_CBANK 0x0a04
#define EIATTR_EXTERNS 0x0f04
#define EIATTR_REQNTID 0x1004
#define EIATTR_FRAME_SIZE 0x1104
#define EIATTR_MIN_STACK_SIZE 0x1204
#define EIATTR_BINDLESS_TEXTURE_BANK 0x1502
#define EIATTR_BINDLESS_SURFACE_BANK 0x1602
#define EIATTR_KPARAM_INFO 0x1704
#define EIATTR_CBANK_PARAM_SIZE 0x1903
#define EIATTR_MAXREG_COUNT 0x1b03
#define EIATTR_EXIT_INSTR_OFFSETS 0x1c04
#define EIATTR_S2RCTAID_INSTR_OFFSETS 0x1d04
#define EIATTR_CRS_STACK_SIZE 0x1e04
#define EIATTR_NEED_CNP_WRAPPER 0x1f01
#define EIATTR_NEED_CNP_PATCH 0x2001
#define EIATTR_EXPLICIT_CACHING 0x2101
#define EIATTR_MAX_STACK_SIZE 0x2304
#define EIATTR_LD_CACHEMOD_INSTR_OFFSETS 0x2504
#define EIATTR_ATOM_SYS_INSTR_OFFSETS 0x2704
#define EIATTR_COOP_GROUP_INSTR_OFFSETS 0x2804
#define EIATTR_SW1850030_WAR 0x2a01
#define EIATTR_WMMA_USED 0x2b01
#define EIATTR_ATOM16_EMUL_INSTR_REG_MAP 0x2e04
#define EIATTR_REGCOUNT 0x2f04
#define EIATTR_SW2393858_WAR 0x3001
#define EIATTR_INT_WARP_WIDE_INSTR_OFFSETS 0x3104
#define EIATTR_INDIRECT_BRANCH_TARGETS 0x3404
#define EIATTR_SW2861232_WAR 0x3501
#define EIATTR_SW_WAR 0x3604
#define EIATTR_CUDA_API_VERSION 0x3704

typedef struct section_entry_ {
	uint16_t type;
	uint16_t size;
} section_entry_t;

typedef struct const_entry {
	uint32_t sym_idx;
	uint16_t base;
	uint16_t size;
} const_entry_t;

typedef struct func_entry {
	uint32_t sym_idx;
	uint32_t local_size;
} func_entry_t;

typedef struct param_entry {
	uint32_t pad; /* always -1 */
	uint16_t idx;
	uint16_t offset;
	uint32_t size;
} param_entry_t;

typedef struct stack_entry {
	uint16_t size;			
	uint16_t unk16;
	uint32_t unk32;
} stack_entry_t;

typedef struct crs_stack_size_entry {
	uint32_t size;
} crs_stack_size_entry_t;

typedef struct symbol_entry {
	uint64_t offset; /* offset in relocation (c14) */
	uint32_t unk32;
	uint32_t sym_idx;
} symbol_entry_t;

#define GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT 10
#define gdev_cuda_raw_func CUfunc_st
struct gdev_cuda_raw_func {
	char *name;
	void *code_buf;
	uint32_t code_size;
	struct {
		void *buf;
		uint32_t size;
	} cmem[GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT]; /* local to functions. */
	uint32_t reg_count;
	uint32_t bar_count;
	uint32_t stack_depth;
	uint32_t stack_size;
	uint32_t shared_size;
	uint32_t param_base;
	uint32_t param_size;
	uint32_t param_count;
	struct gdev_cuda_param *param_data;
	uint32_t local_size;
	uint32_t local_size_neg;
};

struct gdev_cuda_param {
	int idx;
	uint32_t offset;
	uint32_t size;
	uint32_t flags;
	struct gdev_cuda_param *next;
};

static struct CUfunc_st* malloc_func_if_necessary(struct CUmod_st *mod, const char *name)
{
	struct CUfunc_st *func = malloc(sizeof(struct gdev_cuda_raw_func));
	func->name = name;
	return func;
}

/* prototype definition. */
static int cubin_func_type
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func);

static void cubin_func_skip(char **pos, section_entry_t *e)
{
	*pos += sizeof(section_entry_t);
/*#define GDEV_DEBUG*/
#ifdef GDEV_DEBUG
	printf("/* nv.info: ignore entry type: 0x%04x, size=0x%x */\n",
		   e->type, e->size);
#ifndef __KERNEL__
	if (e->size % 4 == 0) {
		int i;
		for (i = 0; i < e->size / 4; i++) {
			uint32_t val = ((uint32_t*)*pos)[i];
			printf("0x%04x\n", val);
		}
	}
	else {
		int i;
		for (i = 0; i < e->size; i++) {
			unsigned char val = ((unsigned char*)*pos)[i];
			printf("0x%02x\n", (uint32_t)val);
		}
	}
#endif
#endif
	*pos += e->size;
}

static void cubin_func_unknown(char **pos, section_entry_t *e)
{
	GDEV_PRINT("/* nv.info: unknown entry type: 0x%.4x, size=0x%x */\n",
			   e->type, e->size);
	cubin_func_skip(pos, e);
}

static int cubin_func_0a04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	const_entry_t *ce;

	*pos += sizeof(section_entry_t);
	ce = (const_entry_t *)*pos;
	raw_func->param_base = ce->base;
	raw_func->param_size = ce->size;
	*pos += e->size;

	return 0;
}

static int cubin_func_0c04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	*pos += sizeof(section_entry_t);
	/* e->size is a parameter size, but how can we use it here? */
	*pos += e->size;

	return 0;
}

static int cubin_func_0d04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	stack_entry_t *se;

	*pos += sizeof(section_entry_t);
	se = (stack_entry_t*) *pos;
	raw_func->stack_depth = se->size;
	/* what is se->unk16 and se->unk32... */

	*pos += e->size;

	return 0;
}

static int cubin_func_1704
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	param_entry_t *pe;
	struct gdev_cuda_param *param_data;

	*pos += sizeof(section_entry_t);
	pe = (param_entry_t *)*pos;

	param_data = (struct gdev_cuda_param *)MALLOC(sizeof(*param_data));
	param_data->idx = pe->idx;
	param_data->offset = pe->offset;
	param_data->size = pe->size >> 18;
	param_data->flags = pe->size & 0x2ffff;
	
	/* append to the head of the parameter data list. */
	param_data->next = raw_func->param_data;
	raw_func->param_data = param_data;

	*pos += e->size;

	return 0;
}

static int cubin_func_1903
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	int ret;
	char *pos2;

	*pos += sizeof(section_entry_t);
	pos2 = *pos;

	/* obtain parameters information. is this really safe? */
	do {
		section_entry_t *sh_e = (section_entry_t *)pos2;
		ret = cubin_func_1704(&pos2, sh_e, raw_func);
		if (ret)
			return ret;
		raw_func->param_count++;
	} while (((section_entry_t *)pos2)->type == 0x1704);

	/* just check if the parameter size matches. */
	if (raw_func->param_size != e->size) {
		if (e->type == 0x1803) { /* sm_13 needs to set param_size here. */
			raw_func->param_size = e->size;
		}
		else {
			GDEV_PRINT("Parameter size mismatched\n");
			GDEV_PRINT("0x%x and 0x%x\n", raw_func->param_size, e->size);
		}
	}

	*pos = pos2; /* need to check if this is correct! */

	return 0;
}

static int cubin_func_1e04
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	raw_func->stack_size = crse->size << 4;

	*pos += e->size;

	return 0;
}

static int cubin_func_maxreg_count
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	GDEV_PRINT("func maxreg_count: %d, ignored\n", crse->size);

	*pos += e->size;

	return 0;
}

static int cubin_func_sw_war
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	GDEV_PRINT("func sw_war: %d, ignored\n", crse->size);

	*pos += e->size;

	return 0;
}

static int cubin_func_api_version
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	GDEV_PRINT("func api_version: %d, ignored\n", crse->size);

	*pos += e->size;

	return 0;
}

static int cubin_func_regcount
(char **pos, section_entry_t *e, struct gdev_kernel *k) 
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	// k->reg_count = crse->size;
	GDEV_PRINT("reg count %d, ignored\n", crse->size);

	*pos += e->size;

	return 0;
}

static int cubin_func_max_stack
(char **pos, section_entry_t *e, struct gdev_kernel *k) 
{
	crs_stack_size_entry_t *crse;

	*pos += sizeof(section_entry_t);
	crse = (crs_stack_size_entry_t*) *pos;
	// k->reg_count = crse->size;
	GDEV_PRINT("max stack %d, ignored\n", crse->size);

	*pos += e->size;

	return 0;
}

static int cubin_func_type
(char **pos, section_entry_t *e, struct gdev_cuda_raw_func *raw_func)
{
	switch (e->type) {
	case 0x0204: /* textures */
		cubin_func_skip(pos, e);
		break;
	case 0x0a04: /* kernel parameters base and size */
		return cubin_func_0a04(pos, e, raw_func);
	case 0x0b04: /* 4-byte align data relevant to params (sm_13) */
	case 0x0c04: /* 4-byte align data relevant to params (sm_20) */
		return cubin_func_0c04(pos, e, raw_func);
	case 0x0d04: /* stack information, hmm... */
		return cubin_func_0d04(pos, e, raw_func);
	case 0x1104: /* ignore recursive call */
		cubin_func_skip(pos, e);
		break;
	case 0x1204: /* some counters but what is this? */
		cubin_func_skip(pos, e);
		break;
	case 0x1803: /* kernel parameters itself (sm_13) */
	case 0x1903: /* kernel parameters itself (sm_20/sm_30) */
		return cubin_func_1903(pos, e, raw_func);
	case 0x1704: /* each parameter information */
		return cubin_func_1704(pos, e, raw_func);
	case 0x1e04: /* crs stack size information */
		return cubin_func_1e04(pos, e, raw_func);
	case 0x0001: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0x080d: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0xf000: /* maybe just padding??? */
		*pos += 4;
		break;
	case 0xffff: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case 0x0020: /* ??? */
		cubin_func_skip(pos, e);
		break;
	case EIATTR_MAXREG_COUNT:
		return cubin_func_maxreg_count(pos, e, raw_func);
	case EIATTR_SW_WAR:
		return cubin_func_sw_war(pos, e, raw_func);
	case EIATTR_CUDA_API_VERSION:
		return cubin_func_api_version(pos, e, raw_func);
	default: /* real unknown */
		cubin_func_unknown(pos, e);
		/* return -EINVAL; */
	}

	return 0;
}

static uint32_t sm_2_arch(uint32_t sm) {
	uint32_t base = sm / 10;
	switch (base) {
		case 2:
			// Fermi: sm_20
			return NV_CHIPSET_FERMI;
		case 3:
			// Kepler: sm_30, sm_35, sm_37
			return NV_CHIPSET_KEPLER;
		case 5:
			// Maxwell: sm_50, sm_52, sm_53
			return NV_CHIPSET_MAXWELL;
		case 6:
			// Pascal: sm_60, sm_61, sm_62
			return NV_CHIPSET_PASCAL;
		case 7:
			// Volta:  sm_70, sm_72
			// Turing: sm_75
			return (sm >= 75)? NV_CHIPSET_TURING : NV_CHIPSET_VOLTA;
		default:
			return 0;
	}
}

static int load_cubin(struct CUmod_st *mod, char *bin)
{
	Elf_Ehdr *ehead;
	Elf_Shdr *sheads;
	Elf_Phdr *pheads;
	Elf_Sym *symbols, *sym;
	char *strings;
	char *shstrings;
	char *nvinfo, *nvrel, *nvglobal_init;
	uint32_t symbols_size, flags;
	int symbols_idx, strings_idx;
	int nvinfo_idx, nvrel_idx, nvrel_const_idx,	nvglobal_idx, nvglobal_init_idx;
	symbol_entry_t *sym_entry;
	section_entry_t *se;
	void *sh;
	char *sh_name;
	char *pos;
	int i, ret = 0;

	if (memcmp(bin, "\177ELF", 4))
		return -ENOENT;

	/* initialize ELF variables. */
	ehead = (Elf_Ehdr *)bin;
	sheads = (Elf_Shdr *)(bin + ehead->e_shoff);
	pheads = (Elf_Phdr *)(bin + ehead->e_phoff);
	flags  = ehead->e_flags;
	symbols = NULL;
	strings = NULL;
	nvinfo = NULL;
	nvrel = NULL;
	nvglobal_init = NULL;
	symbols_idx = 0;
	strings_idx = 0;
	nvinfo_idx = 0;
	nvrel_idx = 0;
	nvrel_const_idx = 0;
	nvglobal_idx = 0;
	nvglobal_init_idx = 0;
	shstrings = bin + sheads[ehead->e_shstrndx].sh_offset;

	// set arch version
	GDEV_PRINT("version is sm_%d, flag %lx\n", flags & 0xff, flags);
	sm_2_arch(flags & 0xff);

	/* seek the ELF header. */
	for (i = 0; i < ehead->e_shnum; i++) {
		sh_name = (char *)(shstrings + sheads[i].sh_name);
		sh = bin + sheads[i].sh_offset;
		/* the following are function-independent sections. */
		switch (sheads[i].sh_type) {
		case SHT_SYMTAB: /* symbol table */
			symbols_idx = i;
			symbols = (Elf_Sym *)sh;
			break;
		case SHT_STRTAB: /* string table */
			strings_idx = i;
			strings = (char *)sh;
			break;
		case SHT_REL: /* relocatable: not sure if nvcc uses it... */
			nvrel_idx = i;
			nvrel = (char *)sh;
			sscanf(sh_name, "%*s%d", &nvrel_const_idx);
			break;
		default:
			/* we never know what sections (.text.XXX, .info.XXX, etc.)
			   appears first for each function XXX... */
			if (!strncmp(sh_name, SH_TEXT, strlen(SH_TEXT))) {
				struct CUfunc_st *func = NULL;
				struct gdev_cuda_raw_func *raw_func = NULL;

				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_TEXT));
				if (!func)
					goto fail_malloc_func;

				raw_func = (struct gdev_cuda_raw_func*)func;

				/* basic information. */
				raw_func->code_buf = bin + sheads[i].sh_offset; /* ==sh */
				raw_func->code_size = sheads[i].sh_size;
				raw_func->reg_count = (sheads[i].sh_info >> 24) & 0x3f;
				raw_func->bar_count = (sheads[i].sh_flags >> 20) & 0xf;
			}
			else if (!strncmp(sh_name, SH_CONST, strlen(SH_CONST))) {
				char fname[256] = {0};
				int x; /* cX[] */
				sscanf(sh_name, SH_CONST "%d.%s", &x, fname);
				/* global constant spaces. */
				if (strlen(fname) == 0) {
					// mod->cmem[x].buf = bin + sheads[i].sh_offset;
					// mod->cmem[x].raw_size = sheads[i].sh_size;
				}
				else if (x >= 0 && x < GDEV_NVIDIA_CONST_SEGMENT_MAX_COUNT) {
					struct CUfunc_st *func = NULL;
					/* this function does nothing if func is already allocated. */
					func = malloc_func_if_necessary(mod, fname);
					if (!func)
						goto fail_malloc_func;
					func->cmem[x].buf = bin + sheads[i].sh_offset;
					func->cmem[x].size = sheads[i].sh_size;
				}
			}
			else if (!strncmp(sh_name, SH_SHARED, strlen(SH_SHARED))) {
				struct CUfunc_st *func = NULL;
				/* this function does nothing if func is already allocated. */
				func =  malloc_func_if_necessary(mod, sh_name + strlen(SH_SHARED));
				if (!func)
					goto fail_malloc_func;
				func->shared_size = sheads[i].sh_size;
				/*
				 * int x;
				 * for (x = 0; x < raw_func->shared_size/4; x++) {
				 * 		unsigned long *data = bin + sheads[i].sh_offset;
				 *		printf("0x%x: 0x%x\n", x*4, data[x]);
				 * }
				 */
			}
			else if (!strncmp(sh_name, SH_LOCAL, strlen(SH_LOCAL))) {
				struct CUfunc_st *func = NULL;
				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_LOCAL));
				if (!func)
					goto fail_malloc_func;
				func->local_size = sheads[i].sh_size;
				func->local_size_neg = 0x7c0; /* FIXME */
			}
			/* NOTE: there are two types of "info" sections: 
			   1. ".nv.info.funcname"
			   2. ".nv.info"
			   ".nv.info.funcname" represents function information while 
			   ".nv.info" points to all ".nv.info.funcname" sections and
			   provide some global data information.
			   NV50 doesn't support ".nv.info" section. 
			   we also assume that ".nv.info.funcname" is an end mark. */
			else if (!strncmp(sh_name, SH_INFO_FUNC, strlen(SH_INFO_FUNC))) {
				struct CUfunc_st *func = NULL;
				struct gdev_cuda_raw_func *raw_func = NULL;
				/* this function does nothing if func is already allocated. */
				func = malloc_func_if_necessary(mod, sh_name + strlen(SH_INFO_FUNC));
				if (!func)
					goto fail_malloc_func;

				raw_func = (struct gdev_cuda_raw_func*)func;

				/* look into the nv.info.@raw_func->name information. */
				pos = (char *) sh;
				while (pos < (char *) sh + sheads[i].sh_size) {
					se = (section_entry_t*) pos;
					ret = cubin_func_type(&pos, se, raw_func);
					if (ret)
						goto fail_cubin_func_type;
				}
			}
			else if (!strcmp(sh_name, SH_INFO)) {
				nvinfo_idx = i;
				nvinfo = (char *) sh;
			}
			else if (!strcmp(sh_name, SH_GLOBAL)) {
				/* symbol space size. */
				symbols_size = sheads[i].sh_size;
				nvglobal_idx = i;
			}
			else if (!strcmp(sh_name, SH_GLOBAL_INIT)) {
				nvglobal_init_idx = i;
				nvglobal_init = (char *) sh;
			}
			break;
		}
	}

	/* nv.rel... "__device__" symbols? */
	for (sym_entry = (symbol_entry_t *)nvrel; 
		 (void *)sym_entry < (void *)nvrel + sheads[nvrel_idx].sh_size;
		 sym_entry++) {
		/*
		 char *sym_name, *sh_name;
		 uint32_t size;
		 sym  = &symbols[se->sym_idx];
		 sym_name = strings + sym->st_name;
		 sh_name = strings + sheads[sym->st_shndx].sh_name;
		 size = sym->st_size;
		*/
	}

	/* symbols: __constant__ variable and built-in function names. */
	for (sym = &symbols[0]; 
		 (void *)sym < (void *)symbols + sheads[symbols_idx].sh_size; sym++) {
		 char *sym_name = strings + sym->st_name;
		 char *sh_name = shstrings + sheads[sym->st_shndx].sh_name;
		 switch (sym->st_info) {
		 case 0x0: /* ??? */
			 break;
		 case 0x2: /* ??? */
			 break;
		 case 0x3: /* ??? */
			 break;
		 case 0x1:
		 case 0x11: /* __device__/__constant__ symbols */
			 if (sym->st_shndx == nvglobal_idx) { /* __device__ */
			 }
			 else { /* __constant__ */
				// TODO: constant
			 }
			 break;
		 case 0x12: /* function symbols */
			 break;
		 case 0x22: /* quick hack: FIXME! */
			 GDEV_PRINT("sym_name: %s\n", sym_name);
			 GDEV_PRINT("sh_name: %s\n", sh_name);
			 GDEV_PRINT("st_value: 0x%llx\n", (unsigned long long)sym->st_value);
			 GDEV_PRINT("st_size: 0x%llx\n", (unsigned long long)sym->st_size);
			 break;
		 default: /* ??? */
			 GDEV_PRINT("/* unknown symbols: 0x%x\n */", sym->st_info);
			 goto fail_symbol;
		 }
	}
	if (nvinfo) { /* >= sm_20 */
		/* parse nv.info sections. */
		pos = (char*)nvinfo;
		while (pos < nvinfo + sheads[nvinfo_idx].sh_size) {
			section_entry_t *e = (section_entry_t*) pos;
			switch (e->type) {
			case 0x0704: /* texture */
				cubin_func_skip(&pos, e);
				break;
			case 0x1104:  /* function */
				cubin_func_skip(&pos, e);
				break;
			case 0x1204: /* some counters but what is this? */
				cubin_func_skip(&pos, e);
				break;
			case EIATTR_REGCOUNT:
				cubin_func_regcount(&pos, e, NULL);
				break;
			case EIATTR_MAX_STACK_SIZE:
				cubin_func_max_stack(&pos, e, NULL);
				break;
			default:
				cubin_func_unknown(&pos, e);
				/* goto fail_function; */
			}
		}
	}

	return 0;

fail_symbol:
fail_cubin_func_type:
fail_malloc_func:
	fprintf(stderr, "error in parsing\n");

	return ret;
}

int init_cubin_image(void* cubin_ptr) {
	return load_cubin((struct CUmod_st *)0, (char*)cubin_ptr);
}