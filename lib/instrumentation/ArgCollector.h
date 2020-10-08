//
// Created by ahueck on 08.10.20.
//

#ifndef TYPEART_ARGCOLLECTOR_H
#define TYPEART_ARGCOLLECTOR_H

#include "Instrumentation.h"

namespace typeart {
class TypeManager;
class InstrumentationHelper;

class MemOpArgCollector final : public ArgumentCollector {
  TypeManager& tm;
  InstrumentationHelper& instr;

 public:
  MemOpArgCollector(TypeManager&, InstrumentationHelper&);
  HeapArgList collectHeap(const llvm::SmallVectorImpl<MallocData>& mallocs) override;
  FreeArgList collectFree(const llvm::SmallVectorImpl<FreeData>& frees) override;
  StackArgList collectStack(const llvm::SmallVectorImpl<AllocaData>& allocs) override;
  GlobalArgList collectGlobal(const llvm::SmallVectorImpl<GlobalData>& globals) override;
  virtual ~MemOpArgCollector() = default;
};
}  // namespace typeart

#endif  // TYPEART_ARGCOLLECTOR_H
