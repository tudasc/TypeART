// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TypeARTFunctions.h"

#include "support/Logger.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace typeart {
class InstrumentationHelper;
}  // namespace typeart

using namespace llvm;

namespace typeart {

TAFunctionDeclarator::TAFunctionDeclarator(Module& m, InstrumentationHelper&, TAFunctions& tafunc)
    : m(m), tafunc(tafunc) {
}

llvm::Function* TAFunctionDeclarator::make_function(IFunc id, llvm::StringRef basename,
                                                    llvm::ArrayRef<llvm::Type*> args, bool with_omp, bool fixed_name) {
  const auto make_fname = [&fixed_name](llvm::StringRef name, llvm::ArrayRef<llvm::Type*> args, bool with_omp) {
    std::string fname;
    llvm::raw_string_ostream os(fname);
    os << name;

    if (!fixed_name) {
      os << "_" << std::to_string(args.size());
    }
    if (with_omp) {
      os << "_"
         << "omp";
    }
    return os.str();
  };

  const auto name = make_fname(basename, args, with_omp);

  if (auto it = f_map.find(name); it != f_map.end()) {
    return it->second;
  }

  auto& c                           = m.getContext();
  const auto addOptimizerAttributes = [&](llvm::Function* f) {
    for (Argument& arg : f->args()) {
      if (arg.getType()->isPointerTy()) {
        arg.addAttr(Attribute::NoCapture);
        arg.addAttr(Attribute::ReadOnly);
      }
    }
  };
  const auto setFunctionLinkageExternal = [](llvm::Function* f) {
    f->setLinkage(GlobalValue::ExternalLinkage);
    //     f->setLinkage(GlobalValue::ExternalWeakLinkage);
  };
  const auto do_make = [&](auto& name, auto f_type) {
    const bool has_f = m.getFunction(name) != nullptr;
    auto fc          = m.getOrInsertFunction(name, f_type);

    Function* f{nullptr};
    if (has_f) {
      LOG_WARNING("Function " << name << " is already declared in the module.")
      f = dyn_cast<Function>(fc.getCallee()->stripPointerCasts());
    } else {
      f = dyn_cast<Function>(fc.getCallee());
      setFunctionLinkageExternal(f);
    }
    addOptimizerAttributes(f);
    return f;
  };

  auto f = do_make(name, FunctionType::get(Type::getVoidTy(c), args, false));

  f_map[name] = f;

  tafunc.putFunctionFor(id, f);

  return f;
}

const llvm::StringMap<llvm::Function*>& TAFunctionDeclarator::getFunctionMap() const {
  return f_map;
}

TAFunctions::TAFunctions() = default;

Function* TAFunctions::getFunctionFor(IFunc id) {
  return typeart_callbacks[id];
}

void TAFunctions::putFunctionFor(IFunc id, llvm::Function* f) {
  typeart_callbacks[id] = f;
}

}  // namespace typeart
