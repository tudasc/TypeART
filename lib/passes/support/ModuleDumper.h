// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef MODULE_DUMPER_UTIL_H
#define MODULE_DUMPER_UTIL_H

#include "llvm/IR/Module.h"

namespace typeart::util::module {

enum class ModulePhase { kBase, kHeap, kOpt, kStack };

void dump_module(const llvm::Module& module, ModulePhase phase);

}  // namespace typeart::util::module

#endif
