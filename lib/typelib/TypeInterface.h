// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
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
extern "C" {
#endif

typedef enum typeart_builtin_type_t {  // NOLINT
  TYPEART_UNKNOWN_TYPE = 0,

  TYPEART_POINTER,
  TYPEART_VTABLE_POINTER,
  TYPEART_VOID,
  TYPEART_NULLPOINTER,

  TYPEART_BOOL,

  TYPEART_CHAR_8,

  TYPEART_UCHAR_8,

  TYPEART_UTF_CHAR_8,
  TYPEART_UTF_CHAR_16,
  TYPEART_UTF_CHAR_32,

  TYPEART_INT_8,
  TYPEART_INT_16,
  TYPEART_INT_32,
  TYPEART_INT_64,
  TYPEART_INT_128,

  TYPEART_UINT_8,
  TYPEART_UINT_16,
  TYPEART_UINT_32,
  TYPEART_UINT_64,
  TYPEART_UINT_128,

  TYPEART_FLOAT_8,
  TYPEART_FLOAT_16,
  TYPEART_FLOAT_32,
  TYPEART_FLOAT_64,
  TYPEART_FLOAT_128,

  TYPEART_COMPLEX_64,
  TYPEART_COMPLEX_128,
  TYPEART_COMPLEX_256,

  TYPEART_NUM_VALID_IDS,

  TYPEART_NUM_RESERVED_IDS = 256
} typeart_builtin_type;

#ifdef __cplusplus
}
#endif

#endif  // TYPEART_TYPEINTERFACE_H
