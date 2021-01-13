//
// Created by sebastian on 13.01.21.
//

#include "Runtime.h"

#include "llvm/ADT/Optional.h"

#include <iostream>

namespace typeart {

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom) {
  std::string buf;
  llvm::raw_string_ostream s(buf);
  const auto name = runtime.typeResolution.getTypeName(typeId);
  s << memAddr << " " << typeId << " " << name << " " << typeSize << " " << count << " (" << calledFrom << ")";
  return s.str();
}

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom) {
  const auto typeSize = runtime.typeResolution.getTypeSize(typeId);
  return toString(memAddr, typeId, count, typeSize, calledFrom);
}

std::string toString(const void* addr, const PointerInfo& info) {
  return toString(addr, info.typeId, info.count, info.debug);
}

}  // namespace debug

RuntimeSystem::RuntimeSystem() {
  debug::printTraceStart();
}

RuntimeSystem::~RuntimeSystem() {
  std::string stats;
  llvm::raw_string_ostream stream(stats);
  softcounter::serialise(recorder, stream);
  if (!stream.str().empty()) {
    // llvm::errs/LOG will crash with virtual call error
    std::cerr << stream.str();
  }
}

/**
 * The global runtime instance.
 */
RuntimeSystem runtime;

}  // namespace typeart