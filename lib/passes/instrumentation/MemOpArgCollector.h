//
// Created by ahueck on 08.10.20.
//

#ifndef TYPEART_MEMOPARGCOLLECTOR_H
#define TYPEART_MEMOPARGCOLLECTOR_H

#include "Instrumentation.h"

namespace typeart {
class TypeManager;
class InstrumentationHelper;

class MemOpArgCollector final : public ArgumentCollector {
  TypeManager* type_m;
  InstrumentationHelper* instr_helper;

 public:
  MemOpArgCollector(TypeManager&, InstrumentationHelper&);
  HeapArgList collectHeap(const MallocDataList& mallocs) override;
  FreeArgList collectFree(const FreeDataList& frees) override;
  StackArgList collectStack(const AllocaDataList& allocs) override;
  GlobalArgList collectGlobal(const GlobalDataList& globals) override;
  
  // TyCart - BEGIN
  AssertArgList collectAssert(const AssertDataList& asserts) override;
  // TyCart - END
};
}  // namespace typeart

#endif  // TYPEART_MEMOPARGCOLLECTOR_H
