// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Instrumentation.h"

#include <utility>

namespace typeart {

InstrumentationContext::InstrumentationContext(std::unique_ptr<ArgumentCollector> col,
                                               std::unique_ptr<MemoryInstrument> instr)
    : collector(std::move(col)), instrumenter(std::move(instr)) {
}

InstrCount InstrumentationContext::handleHeap(const MallocDataList& mallocs) {
  if (mallocs.empty()) {
    return 0;
  }
  auto heap_args        = collector->collectHeap(mallocs);
  const auto heap_count = instrumenter->instrumentHeap(heap_args);
  return heap_count;
}

InstrCount InstrumentationContext::handleFree(const FreeDataList& frees) {
  if (frees.empty()) {
    return 0;
  }
  auto free_args        = collector->collectFree(frees);
  const auto free_count = instrumenter->instrumentFree(free_args);
  return free_count;
}

InstrCount InstrumentationContext::handleStack(const AllocaDataList& allocas) {
  if (allocas.empty()) {
    return 0;
  }
  auto alloca_args       = collector->collectStack(allocas);
  const auto stack_count = instrumenter->instrumentStack(alloca_args);
  return stack_count;
}

InstrCount InstrumentationContext::handleGlobal(const GlobalDataList& globals) {
  if (globals.empty()) {
    return 0;
  }
  auto global_args        = collector->collectGlobal(globals);
  const auto global_count = instrumenter->instrumentGlobal(global_args);
  return global_count;
}

}  // namespace typeart