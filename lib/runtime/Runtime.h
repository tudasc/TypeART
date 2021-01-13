//
// Created by sebastian on 13.01.21.
//

#ifndef TYPEART_RUNTIME_H
#define TYPEART_RUNTIME_H

#include "AccessCountPrinter.h"
#include "AccessCounter.h"
#include "AllocationTracking.h"
#include "TypeResolution.h"
#include "support/Logger.h"

namespace typeart {

template <typename T>
inline const void* addByteOffset(const void* addr, T offset) {
  return static_cast<const void*>(static_cast<const uint8_t*>(addr) + offset);
}

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom);

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom);

std::string toString(const void* addr, const PointerInfo& info);

inline void printTraceStart() {
  LOG_TRACE("TypeART Runtime Trace");
  LOG_TRACE("*********************");
  LOG_TRACE("Operation  Address   Type   Size   Count   (CallAddr)   Stack/Heap/Global");
  LOG_TRACE("-------------------------------------------------------------------------");
}

}  // namespace debug

struct RuntimeSystem {
  RuntimeSystem();

  ~RuntimeSystem();

  Recorder recorder;
  TypeResolution typeResolution;

  // Global allocation data
  // TODO: Add mutex
  AllocationTracker allocTracker;
};

extern RuntimeSystem runtime;

}  // namespace typeart

#endif  // TYPEART_RUNTIME_H
