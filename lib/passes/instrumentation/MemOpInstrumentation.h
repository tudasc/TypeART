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

#ifndef TYPEART_MEMOPINSTRUMENTATION_H
#define TYPEART_MEMOPINSTRUMENTATION_H

#include "Instrumentation.h"

namespace typeart {

class TAFunctionQuery;
class InstrumentationHelper;

class MemOpInstrumentation final : public MemoryInstrument {
  TAFunctionQuery* fquery;
  InstrumentationHelper* instr_helper;
  bool instrument_lifetime{false};

 public:
  MemOpInstrumentation(TAFunctionQuery& fquery, InstrumentationHelper& instr, bool lifetime_instrument = false);
  InstrCount instrumentHeap(const HeapArgList& heap) override;
  InstrCount instrumentFree(const FreeArgList& frees) override;
  InstrCount instrumentStack(const StackArgList& stack) override;
  InstrCount instrumentGlobal(const GlobalArgList& globals) override;
};

}  // namespace typeart
#endif  // TYPEART_MEMOPINSTRUMENTATION_H
