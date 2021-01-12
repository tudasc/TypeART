//
// Created by sebastian on 08.01.21.
//

#include "RuntimeUtils.h"
#include "TypeResolution.h"

namespace typeart {

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom) {
  std::string buf;
  llvm::raw_string_ostream s(buf);
  const auto name = TypeResolution::get().getTypeName(typeId);
  s << memAddr << " " << typeId << " " << name << " " << typeSize << " " << count << " (" << calledFrom << ")";
  return s.str();
}

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom) {
  const auto typeSize = TypeResolution::get().getTypeSize(typeId);
  return toString(memAddr, typeId, count, typeSize, calledFrom);
}

std::string toString(const void* addr, const PointerInfo& info) {
  return toString(addr, info.typeId, info.count, info.debug);
}

}
}
