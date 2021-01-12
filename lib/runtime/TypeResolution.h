//
// Created by sebastian on 11.01.21.
//

#ifndef TYPEART_TYPERESOLUTION_H
#define TYPEART_TYPERESOLUTION_H

#include "RuntimeInterface.h"
#include "TypeDB.h"

namespace typeart {


class TypeResolution {
  TypeDB typeDB;

  TypeResolution();

 public:
  using TypeArtStatus = typeart_status;

  static TypeResolution& get() {
    // Note: This is guaranteed to be thread-safe
    static TypeResolution instance;
    return instance;
  }

  size_t getMemberIndex(typeart_struct_layout structInfo, size_t offset) const;

  TypeArtStatus getSubTypeInfo(const void* baseAddr, size_t offset, typeart_struct_layout containerInfo, int* subType,
                               const void** subTypeBaseAddr, size_t* subTypeOffset, size_t* subTypeCount) const;

  TypeArtStatus getSubTypeInfo(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo, int* subType,
                               const void** subTypeBaseAddr, size_t* subTypeOffset, size_t* subTypeCount) const;

  TypeArtStatus getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo, int* type,
                                    size_t* count) const;

  TypeArtStatus getTypeInfo(const void* addr, int* type, size_t* count) const;

  TypeArtStatus getContainingTypeInfo(const void* addr, int* type, size_t* count, const void** baseAddress,
                                      size_t* offset) const;

  TypeArtStatus getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const;

  TypeArtStatus getStructInfo(int id, const StructTypeInfo** structInfo) const;

  const std::string& getTypeName(int id) const;

  size_t getTypeSize(int id) const;

  bool isValidType(int id) const;
};

}

#endif  // TYPEART_TYPERESOLUTION_H
