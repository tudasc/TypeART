//
// Created by ahueck on 09.10.20.
//

#ifndef TYPEART_MEMOPINSTRUMENTATION_H
#define TYPEART_MEMOPINSTRUMENTATION_H

#include "Instrumentation.h"

namespace typeart {

class TAFunctionQuery;
class InstrumentationHelper;

class MemOpInstrumentation final : public MemoryInstrument {
  TAFunctionQuery* fquery;
  InstrumentationHelper* instr_helper;

 public:
  MemOpInstrumentation(TAFunctionQuery& fquery, InstrumentationHelper& instr);
  InstrCount instrumentHeap(const HeapArgList& heap) override;
  InstrCount instrumentFree(const FreeArgList& frees) override;
  InstrCount instrumentStack(const StackArgList& stack) override;
  InstrCount instrumentGlobal(const GlobalArgList& globals) override;
};

}  // namespace typeart
#endif  // TYPEART_MEMOPINSTRUMENTATION_H
