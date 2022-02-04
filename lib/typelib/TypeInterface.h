// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_TYPEINTERFACE_H
#define TYPEART_TYPEINTERFACE_H

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum typeart_builtin_type_t {                  // NOLINT
  TYPEART_INT8             = 0,                        //  8 bit signed integer
  TYPEART_INT16            = 1,                        // 16 bit signed integer
  TYPEART_INT32            = 2,                        // 32 bit signed integer
  TYPEART_INT64            = 3,                        // 64 bit signed integer
  TYPEART_HALF             = 4,                        // IEEE 754 half precision floating point type
  TYPEART_FLOAT            = 5,                        // IEEE 754 single precision floating point type
  TYPEART_DOUBLE           = 6,                        // IEEE 754 double precision floating point type
  TYPEART_FP128            = 7,                        // IEEE 754 quadruple precision floating point type
  TYPEART_X86_FP80         = 8,                        // x86 extended precision 80-bit floating point type
  TYPEART_PPC_FP128        = 9,                        // ICM extended precision 128-bit floating point type
  TYPEART_POINTER          = 10,                       // Represents all pointer types
  TYPEART_NUM_VALID_IDS    = TYPEART_POINTER + 1,      // Number of valid built-in types
  TYPEART_UNKNOWN_TYPE     = 255,                      // Placeholder for unknown types
  TYPEART_NUM_RESERVED_IDS = TYPEART_UNKNOWN_TYPE + 1  // Represents user-defined types
} typeart_builtin_type;

#ifdef __cplusplus
}
#endif

#endif  // TYPEART_TYPEINTERFACE_H
