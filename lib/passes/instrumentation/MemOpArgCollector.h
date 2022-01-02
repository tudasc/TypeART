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

#ifndef TYPEART_MEMOPARGCOLLECTOR_H
#define TYPEART_MEMOPARGCOLLECTOR_H

#include "Instrumentation.h"
#include "analysis/MemOpData.h"

namespace typeart {
class TypeGenerator;
class InstrumentationHelper;

class MemOpArgCollector final : public ArgumentCollector {
  TypeGenerator* type_m;
  InstrumentationHelper* instr_helper;

 public:
  MemOpArgCollector(TypeGenerator*, InstrumentationHelper&);
  HeapArgList collectHeap(const MallocDataList& mallocs) override;
  FreeArgList collectFree(const FreeDataList& frees) override;
  StackArgList collectStack(const AllocaDataList& allocs) override;
  GlobalArgList collectGlobal(const GlobalDataList& globals) override;
};
}  // namespace typeart

#endif  // TYPEART_MEMOPARGCOLLECTOR_H
