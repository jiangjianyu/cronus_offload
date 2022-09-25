/*
 * Copyright (C) Shinpei Kato
 *
 * University of California, Santa Cruz
 * Systems Research Lab.
 *
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

#ifndef __GDEV_API_H__
#define __GDEV_API_H__

#define NV_CHIPSET_MASK 0x1f0

/* The following macro id is from the kernel tree 
 * linux/drivers/gpu/drm/nouveau/include/nvif/class.h
 */

#define NV_CHIPSET_NV50     0x50
#define NV_CHIPSET_FERMI    0xC0
#define NV_CHIPSET_KEPLER   0xE0
#define NV_CHIPSET_KEPLER_B 0xF0
#define NV_CHIPSET_MAXWELL  0x110
#define NV_CHIPSET_PASCAL   0x130
#define NV_CHIPSET_VOLTA    0x140
#define NV_CHIPSET_TURING   0x160
#define NV_CHIPSET_AMPERE   0x170


#define NV50_MEMORY_TO_MEMORY_FORMAT                                 0x00005039
#define FERMI_MEMORY_TO_MEMORY_FORMAT                                0x00009039
#define KEPLER_MEMORY_TO_MEMORY_FORMAT                               0x0000a040
#define KEPLER_B_MEMORY_TO_MEMORY_FORMAT                             0x0000a140

#define NV50_COMPUTE_A                                                 0x000050c0
#define FERMI_COMPUTE_A                                                0x000090c0
#define FERMI_COMPUTE_B                                                0x000091c0
#define KEPLER_COMPUTE_A                                               0x0000a0c0
#define KEPLER_COMPUTE_B                                               0x0000a1c0
#define MAXWELL_COMPUTE_A                                              0x0000b0c0
#define MAXWELL_COMPUTE_B                                              0x0000b1c0
#define PASCAL_COMPUTE_A                                               0x0000c0c0
#define PASCAL_COMPUTE_B                                               0x0000c1c0
#define VOLTA_COMPUTE_A                                                0x0000c3c0
#define TURING_COMPUTE_A                                               0x0000c5c0

#endif
