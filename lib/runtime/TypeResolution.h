//
// Created by sebastian on 11.01.21.
//

#ifndef TYPEART_TYPERESOLUTION_H
#define TYPEART_TYPERESOLUTION_H

#include "AccessCounter.h"
#include "RuntimeData.h"
#include "RuntimeInterface.h"
#include "TypeDB.h"

namespace typeart {

class TypeResolution {
  const TypeDB& typeDB;
  Recorder& recorder;

 public:
  using TypeArtStatus = typeart_status;

  TypeResolution(const TypeDB& db, Recorder& recorder);

  size_t getMemberIndex(typeart_struct_layout structInfo, size_t offset) const;

  TypeArtStatus getSubTypeInfo(const void* baseAddr, size_t offset, typeart_struct_layout containerInfo, int* subType,
                               const void** subTypeBaseAddr, size_t* subTypeOffset, size_t* subTypeCount) const;

  TypeArtStatus getSubTypeInfo(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo, int* subType,
                               const void** subTypeBaseAddr, size_t* subTypeOffset, size_t* subTypeCount) const;

  TypeArtStatus getTypeInfoInternal(const void* baseAddr, size_t offset, const StructTypeInfo& containerInfo, int* type,
                                    size_t* count) const;

  TypeArtStatus getTypeInfo(const void* addr, const void* basePtr, const PointerInfo& ptrInfo, int* type,
                            size_t* count) const;

  TypeArtStatus getContainingTypeInfo(const void* addr, const void* basePtr, const PointerInfo& ptrInfo, size_t* count,
                                      size_t* offset) const;

  TypeArtStatus getBuiltinInfo(const void* addr, const PointerInfo& ptrInfo, typeart::BuiltinType* type) const;

  TypeArtStatus getStructInfo(int id, const StructTypeInfo** structInfo) const;

  const std::string& getTypeName(int id) const;

  size_t getTypeSize(int id) const;

  bool isValidType(int id) const;
};

}  // namespace typeart

#endif  // TYPEART_TYPERESOLUTION_H
