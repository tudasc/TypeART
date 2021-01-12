//
// Created by sebastian on 07.01.21.
//

#include "TypeResolution.h"

#include "AccessCountPrinter.h"
#include "AccessCounter.h"
#include "AllocationTracking.h"
#include "TypeDB.h"
#include "TypeIO.h"
#include "RuntimeUtils.h"

namespace typeart {

static constexpr const char* defaultTypeFileName = "types.yaml";

TypeResolution::TypeResolution() {
  // TODO: Init recorder here or in alloc tracker?
  // This keeps the destruction order: 1. RT 2. Recorder
  auto& recorder = Recorder::get();

  auto loadTypes = [this](const std::string& file) -> bool {
    TypeIO cio(typeDB);
    return cio.load(file);
  };

  // Try to load types from specified file first.
  // Then look at default location.
  const char* typeFile = std::getenv("TA_TYPE_FILE");
  if (typeFile != nullptr) {
    if (!loadTypes(typeFile)) {
      LOG_FATAL("Failed to load recorded types from " << typeFile);
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  } else {
    if (!loadTypes(defaultTypeFileName)) {
      LOG_FATAL(
          "No type file with default name \""
              << defaultTypeFileName
              << "\" in current directory. To specify a different file, edit the TA_TYPE_FILE environment variable.");
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  }

  std::stringstream ss;
  const auto& typeList = typeDB.getStructList();
  for (const auto& structInfo : typeList) {
    ss << structInfo.name << ", ";
  }
  recorder.incUDefTypes(typeList.size());
  LOG_INFO("Recorded types: " << ss.str());

}

size_t TypeResolution::getMemberIndex(typeart_struct_layout structInfo, size_t offset) const {
  size_t n = structInfo.len;
  if (offset > structInfo.offsets[n - 1]) {
    return n - 1;
  }

  size_t i = 0;
  while (i < n - 1 && offset >= structInfo.offsets[i + 1]) {
    ++i;
  }
  return i;
}

TypeResolution::TypeArtStatus TypeResolution::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                   typeart_struct_layout containerInfo, int* subType,
                                                   const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                   size_t* subTypeCount) const {
  if (offset >= containerInfo.extent) {
    return TA_BAD_OFFSET;
  }

  // Get index of the struct member at the address
  size_t memberIndex = getMemberIndex(containerInfo, offset);

  int memberType = containerInfo.member_types[memberIndex];

  size_t baseOffset = containerInfo.offsets[memberIndex];
  assert(offset >= baseOffset && "Invalid offset values");

  size_t internalOffset   = offset - baseOffset;
  size_t typeSize         = typeDB.getTypeSize(memberType);
  size_t offsetInTypeSize = internalOffset / typeSize;
  size_t newOffset        = internalOffset % typeSize;

  // If newOffset != 0, the subtype cannot be atomic, i.e. must be a struct
  if (newOffset != 0) {
    if (typeDB.isReservedType(memberType)) {
      return TA_BAD_ALIGNMENT;
    }
  }

  // Ensure that the array index is in bounds
  if (offsetInTypeSize >= containerInfo.count[memberIndex]) {
    // Address points to padding
    return TA_BAD_ALIGNMENT;
  }

  *subType         = memberType;
  *subTypeBaseAddr = addByteOffset(baseAddr, baseOffset);
  *subTypeOffset   = newOffset;
  *subTypeCount    = containerInfo.count[memberIndex] - offsetInTypeSize;

  return TA_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getSubTypeInfo(const void* baseAddr, size_t offset,
                                                   const StructTypeInfo& containerInfo, int* subType,
                                                   const void** subTypeBaseAddr, size_t* subTypeOffset,
                                                   size_t* subTypeCount) const {
  typeart_struct_layout structLayout;
  structLayout.id           = containerInfo.id;
  structLayout.name         = containerInfo.name.c_str();
  structLayout.len          = containerInfo.numMembers;
  structLayout.extent       = containerInfo.extent;
  structLayout.offsets      = &containerInfo.offsets[0];
  structLayout.member_types = &containerInfo.memberTypes[0];
  structLayout.count        = &containerInfo.arraySizes[0];
  return getSubTypeInfo(baseAddr, offset, structLayout, subType, subTypeBaseAddr, subTypeOffset, subTypeCount);
}

TypeResolution::TypeArtStatus TypeResolution::getTypeInfoInternal(const void* baseAddr, size_t offset,
                                                        const StructTypeInfo& containerInfo, int* type,
                                                        size_t* count) const {
  assert(offset < containerInfo.extent && "Something went wrong with the base address computation");

  TypeArtStatus status;
  int subType;
  const void* subTypeBaseAddr;
  size_t subTypeOffset;
  size_t subTypeCount;
  const StructTypeInfo* structInfo = &containerInfo;

  bool resolve = true;

  // Resolve type recursively, until the address matches exactly
  while (resolve) {
    status = getSubTypeInfo(baseAddr, offset, *structInfo, &subType, &subTypeBaseAddr, &subTypeOffset, &subTypeCount);

    if (status != TA_OK) {
      return status;
    }

    baseAddr = subTypeBaseAddr;
    offset   = subTypeOffset;

    // Continue as long as there is a byte offset
    resolve = offset != 0;

    // Get layout of the nested struct
    if (resolve) {
      status = getStructInfo(subType, &structInfo);
      if (status != TA_OK) {
        return status;
      }
    }
  }
  *type  = subType;
  *count = subTypeCount;
  return TA_OK;
}

TypeResolution::TypeArtStatus TypeResolution::getTypeInfo(const void* addr, int* type, size_t* count) const {
  int containingType;
  size_t containingTypeCount;
  const void* baseAddr;
  size_t internalOffset;

  // First, retrieve the containing type
  TypeArtStatus status = getContainingTypeInfo(addr, &containingType, &containingTypeCount, &baseAddr, &internalOffset);
  if (status != TA_OK) {
    if (status == TA_UNKNOWN_ADDRESS) {
      typeart::Recorder::get().incAddrMissing(addr);
    }
    return status;
  }

  // Check for exact address match
  if (internalOffset == 0) {
    *type  = containingType;
    *count = containingTypeCount;
    return TA_OK;
  }

  if (typeDB.isBuiltinType(containingType)) {
    // Address points to the middle of a builtin type
    return TA_BAD_ALIGNMENT;
  }

  // Resolve struct recursively
  const auto* structInfo = typeDB.getStructInfo(containingType);
  if (structInfo != nullptr) {
    const void* containingTypeAddr = addByteOffset(addr, -internalOffset);
    return getTypeInfoInternal(containingTypeAddr, internalOffset, *structInfo, type, count);
  }
  return TA_INVALID_ID;
}

TypeResolution::TypeArtStatus TypeResolution::getContainingTypeInfo(const void* addr, int* type, size_t* count,
                                                          const void** baseAddress, size_t* offset) const {
  // Find the start address of the containing buffer
  auto ptrData = findBaseAlloc(addr);

  if (ptrData) {
    const auto& basePtrInfo = ptrData.getValue().second;
    const auto* basePtr     = ptrData.getValue().first;
    size_t typeSize         = getTypeSize(basePtrInfo.typeId);

    // Check for exact match -> no further checks and offsets calculations needed
    if (basePtr == addr) {
      *type        = basePtrInfo.typeId;
      *count       = basePtrInfo.count;
      *baseAddress = addr;
      *offset      = 0;
      return TA_OK;
    }

    // The address points inside a known array
    const void* blockEnd = addByteOffset(basePtr, basePtrInfo.count * typeSize);

    // Ensure that the given address is in bounds and points to the start of an element
    if (addr >= blockEnd) {
      const std::ptrdiff_t offset = static_cast<const uint8_t*>(addr) - static_cast<const uint8_t*>(basePtr);
      const auto oob_index        = (offset / typeSize) - basePtrInfo.count + 1;
      LOG_ERROR("Out of bounds for the lookup: (" << debug::toString(addr, basePtrInfo)
                                                  << ") #Elements too far: " << oob_index);
      return TA_UNKNOWN_ADDRESS;
    }

    assert(addr >= basePtr && "Error in base address computation");
    size_t addrDif = reinterpret_cast<size_t>(addr) - reinterpret_cast<size_t>(basePtr);

    // Offset of the pointer w.r.t. the start of the containing type
    size_t internalOffset = addrDif % typeSize;

    // Array index
    size_t typeOffset = addrDif / typeSize;
    size_t typeCount  = basePtrInfo.count - typeOffset;

    // Retrieve and return type information
    *type        = basePtrInfo.typeId;
    *count       = typeCount;
    *baseAddress = basePtr;  // addByteOffset(basePtr, typeOffset * basePtrInfo.typeSize);
    *offset      = internalOffset;
    return TA_OK;
  }
  return TA_UNKNOWN_ADDRESS;
}

TypeResolution::TypeArtStatus TypeResolution::getBuiltinInfo(const void* addr, typeart::BuiltinType* type) const {
  int id;
  size_t count;
  TypeArtStatus result = getTypeInfo(addr, &id, &count);
  if (result == TA_OK) {
    if (typeDB.isReservedType(id)) {
      *type = static_cast<BuiltinType>(id);
      return TA_OK;
    }
    return TA_WRONG_KIND;
  }
  return result;
}

TypeResolution::TypeArtStatus TypeResolution::getStructInfo(int id, const StructTypeInfo** structInfo) const {
  // Requested ID must correspond to a struct
  if (!typeDB.isStructType(id)) {
    return TA_WRONG_KIND;
  }

  const auto* result = typeDB.getStructInfo(id);

  if (result != nullptr) {
    *structInfo = result;
    return TA_OK;
  }
  return TA_INVALID_ID;
}

const std::string& TypeResolution::getTypeName(int id) const {
  return typeDB.getTypeName(id);
}

size_t TypeResolution::getTypeSize(int id) const {
  return typeDB.getTypeSize(id);
}

bool TypeResolution::isValidType(int id) const {
  return typeDB.isValid(id);
}



}