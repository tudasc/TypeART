// TypeART library
//
// Copyright (c) 2017-2026 TypeART Authors
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
#include "configuration/Configuration.h"

#include <memory>

namespace typeart {
namespace config {
class Configuration;
}
class TAFunctionQuery;
class InstrumentationHelper;
class TypeRegistry;
class InstrumentationInserter;

class MemOpInstrumentation final : public MemoryInstrument {
  const config::Configuration& typeart_config;
  TAFunctionQuery* function_query;
  // std::unique_ptr<TypeRegistry> type_id_handler;
  InstrumentationHelper* instrumentation_helper;
  std::unique_ptr<InstrumentationInserter> function_instrumenter_;
  bool instrument_lifetime{false};

 public:
  MemOpInstrumentation(const config::Configuration& typeart_conf, TAFunctionQuery* fquery, InstrumentationHelper& instr,
                       std::unique_ptr<InstrumentationInserter> function_instrumenter);
  InstrCount instrumentHeap(const HeapArgList& heap) override;
  InstrCount instrumentFree(const FreeArgList& frees) override;
  InstrCount instrumentStack(const StackArgList& stack) override;
  InstrCount instrumentGlobal(const GlobalArgList& globals) override;
};

}  // namespace typeart
#endif  // TYPEART_MEMOPINSTRUMENTATION_H
