//
// Created by sebastian on 13.01.21.
//

#ifndef TYPEART_RUNTIME_H
#define TYPEART_RUNTIME_H

#include "AccessCounter.h"
#include "AllocationTracking.h"
#include "TypeDB.h"
#include "TypeResolution.h"
#include "support/Logger.h"

namespace typeart {

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom);

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom);

std::string toString(const void* addr, const PointerInfo& info);

}  // namespace debug

struct RuntimeSystem {
  RuntimeSystem();

  ~RuntimeSystem();

  Recorder recorder;
  TypeResolution typeResolution;
  AllocationTracker allocTracker;

 private:
  TypeDB typeDB;
};

extern RuntimeSystem kRuntimeSystem;

}  // namespace typeart

#endif  // TYPEART_RUNTIME_H
