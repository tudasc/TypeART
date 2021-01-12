//
// Created by sebastian on 11.01.21.
//

#ifndef TYPEART_ALLOCATIONTRACKING_H
#define TYPEART_ALLOCATIONTRACKING_H

#include "RuntimeData.h"


namespace llvm {
    template <typename T>
    class Optional;
}  // namespace llvm

namespace typeart {

    llvm::Optional<RuntimeT::MapEntry> findBaseAlloc(const void* addr);

}

#endif  // TYPEART_ALLOCATIONTRACKING_H
