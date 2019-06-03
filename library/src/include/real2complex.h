/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef REAL_TO_COMPLEX_H
#define REAL_TO_COMPLEX_H

#include "tree_node.h"
#include <unordered_map>

/// RCsimple

// real to complex
void real2complex(const void* data, void* back);
void complex2hermitian(const void* data, void* back);

// complex to real
void complex2real(const void* data, void* back);
void hermitian2complex(const void* data, void* back);

/// New RC
void r2c_1d_post(const void* data, void* back);
void c2r_1d_pre(const void* data, void* back);

/// Wrapper for RCsimple and New RC
void real2complex_pre_process(const void* data, void* back);
void real2complex_post_process(const void* data, void* back);

void complex2real_pre_process(const void* data, void* back);
void complex2real_post_process(const void* data, void* back);

#endif // REAL_TO_COMPLEX_H
