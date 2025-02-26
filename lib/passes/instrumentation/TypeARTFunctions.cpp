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

TAFunctionDeclarator::TAFunctionDeclarator(Module& mod, InstrumentationHelper&, TAFunctions& typeart_funcs)
    : module(mod), typeart_functions(typeart_funcs) {
}

llvm::Function* TAFunctionDeclarator::make_function(IFunc func_id, llvm::StringRef basename,
                                                    llvm::ArrayRef<llvm::Type*> args, bool with_omp, bool fixed_name) {
  const auto make_fname = [&fixed_name](llvm::StringRef name, llvm::ArrayRef<llvm::Type*> callback_arguments,
                                        bool with_omp_postfix) {
    std::string fname;
    llvm::raw_string_ostream os(fname);
    os << name;

    if (!fixed_name) {
      os << "_" << std::to_string(callback_arguments.size());
    }
    if (with_omp_postfix) {
      os << "_"
         << "omp";
    }
    return os.str();
  };

  const auto name = make_fname(basename, args, with_omp);

  if (auto it = function_map.find(name); it != function_map.end()) {
    return it->second;
  }

  auto& c                           = module.getContext();
  const auto addOptimizerAttributes = [&](llvm::Function* function) {
    function->setDoesNotThrow();
    function->setDoesNotFreeMemory();
    function->setDoesNotRecurse();
#if LLVM_VERSION_MAJOR >= 12
    function->setWillReturn();
#endif
    for (Argument& arg : function->args()) {
      if (arg.getType()->isPointerTy()) {
        arg.addAttr(Attribute::NoCapture);
        arg.addAttr(Attribute::ReadOnly);
        arg.addAttr(Attribute::NoFree);
      }
    }
  };
  const auto setFunctionLinkageExternal = [](llvm::Function* function) {
    function->setLinkage(GlobalValue::ExternalLinkage);
    //     f->setLinkage(GlobalValue::ExternalWeakLinkage);
  };
  const auto do_make = [&](auto& function_name, auto function_type) {
    const bool has_func_declared = module.getFunction(function_name) != nullptr;
    auto func_in_module          = module.getOrInsertFunction(function_name, function_type);

    Function* function{nullptr};
    if (has_func_declared) {
      LOG_WARNING("Function " << function_name << " is already declared in the module.")
      function = dyn_cast<Function>(func_in_module.getCallee()->stripPointerCasts());
    } else {
      function = dyn_cast<Function>(func_in_module.getCallee());
      setFunctionLinkageExternal(function);
    }

    addOptimizerAttributes(function);
    return function;
  };

  auto generated_function = do_make(name, FunctionType::get(Type::getVoidTy(c), args, false));

  function_map[name] = generated_function;

  typeart_functions.putFunctionFor(func_id, generated_function);

  return generated_function;
}

const llvm::StringMap<llvm::Function*>& TAFunctionDeclarator::getFunctionMap() const {
  return function_map;
}

TAFunctions::TAFunctions() = default;

Function* TAFunctions::getFunctionFor(IFunc id) {
  return typeart_callbacks[id];
}

void TAFunctions::putFunctionFor(IFunc id, llvm::Function* f) {
  typeart_callbacks[id] = f;
}

}  // namespace typeart
