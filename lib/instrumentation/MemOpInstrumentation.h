//
// Created by ahueck on 09.10.20.
//

#ifndef TYPEART_MEMOPINSTRUMENTATION_H
#define TYPEART_MEMOPINSTRUMENTATION_H

#include "Instrumentation.h"
namespace typeart {
class InstrumentationHelper;

class MemOpInstrumentation final : public MemoryInstrument {
  InstrumentationHelper& instr;

 public:
  MemOpInstrumentation(InstrumentationHelper& instr);
  size_t instrumentHeap(const HeapArgList& heap) override;
  size_t instrumentFree(const FreeArgList& frees) override;
  size_t instrumentStack(const StackArgList& frees) override;
  size_t instrumentGlobal(const GlobalArgList& globals) override;
  ~MemOpInstrumentation() override = default;
};
}  // namespace typeart
#endif  // TYPEART_MEMOPINSTRUMENTATION_H
