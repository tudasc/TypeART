//
// Created by ahueck on 17.01.21.
//

#ifndef TYPEART_OMPUTIL_H
#define TYPEART_OMPUTIL_H

#include "Util.h"
#include "support/Logger.h"

#include "llvm/IR/Function.h"

namespace typeart::util::omp {

inline bool isOmpContext(llvm::Function* f) {
  if (f != nullptr) {
    const auto name_ = demangle(f->getName());
    llvm::StringRef fname(name_);
    return fname.startswith(".omp") || fname.startswith("__kmpc") || fname.startswith("__omp");
  }
  return false;
}

}  // namespace typeart::util::omp

#endif  // TYPEART_OMPUTIL_H
