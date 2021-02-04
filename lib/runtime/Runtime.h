//
// Created by sebastian on 13.01.21.
//

#ifndef TYPEART_RUNTIME_H
#define TYPEART_RUNTIME_H

#include "AccessCounter.h"
#include "AllocationTracking.h"
#include "TypeDB.h"
#include "TypeResolution.h"

namespace typeart {

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom);

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom);

std::string toString(const void* addr, const PointerInfo& info);

}  // namespace debug

struct RuntimeSystem {
 private:
  TypeDB typeDB{};

 public:
  Recorder recorder{};
  TypeResolution typeResolution;
  AllocationTracker allocTracker;

  static RuntimeSystem& get() {
    // As opposed to a global variable, a singleton + instantiation during
    // the first callback/query avoids some problems when
    // preloading (especially with MUST).
    static RuntimeSystem instance;
    return instance;
  }

 private:
  RuntimeSystem();
  ~RuntimeSystem();
};

}  // namespace typeart

#endif  // TYPEART_RUNTIME_H
