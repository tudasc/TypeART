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

#ifndef TYPEART_TYPERESOLUTION_H
#define TYPEART_TYPERESOLUTION_H

#include "AccessCounter.h"
#include "RuntimeData.h"
#include "RuntimeInterface.h"
#include "TypeDB.h"
#include "TypeInterface.h"

#include <cstddef>
#include <string>

namespace typeart {

struct PointerInfo;

class TypeResolution {
  const TypeDB& typeDB;
  Recorder& recorder;

 public:
  using TypeArtStatus = typeart_status;

  TypeResolution(const TypeDB& type_db, Recorder& recorder);

  TypeArtStatus getSubTypeInfo(const void* baseAddr, size_t offset, const typeart_struct_layout& containerInfo,
                               int* subType, const void** subTypeBaseAddr, size_t* subTypeOffset,
                               size_t* subTypeCount) const;

  TypeArtStatus getTypeInfoInternal(const void* struct_type_info, size_t offset, const StructTypeInfo& containerInfo,
                                    int* type, size_t* count) const;

  TypeArtStatus getTypeInfo(const void* addr, const void* basePtr, const PointerInfo& ptrInfo, int* type,
                            size_t* count) const;

  TypeArtStatus getContainingTypeInfo(const void* addr, const void* basePtr, const PointerInfo& ptrInfo, size_t* count,
                                      size_t* offset) const;

  TypeArtStatus getStructInfo(int type_id, const StructTypeInfo** structInfo) const;

  [[nodiscard]] const TypeDB& db() const;
};

}  // namespace typeart

#endif  // TYPEART_TYPERESOLUTION_H
