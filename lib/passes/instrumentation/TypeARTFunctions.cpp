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

#include "TypeARTFunctions.h"

#include "support/CudaUtil.h"
#include "support/Logger.h"
#include "support/OmpUtil.h"
#include "support/Util.h"

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

namespace detail {
std::string get_func_suffix(IFunc id) {
  switch (id) {
    case IFunc::free_cuda:
    case IFunc::heap_cuda:
      return "_cuda";
    case IFunc::free_omp:
    case IFunc::heap_omp:
    case IFunc::stack_omp:
    case IFunc::scope_omp:
      return "_omp";
    default:
      return "";
  }
}

enum class IFuncType : unsigned { standard, omp, cuda };

IFuncType ifunc_type_for(llvm::Function* f) {
  if (f == nullptr) {
    return IFuncType::standard;
  }
  if (cuda::is_cuda_function(*f)) {
    return IFuncType::cuda;
  }
  if (util::omp::isOmpContext(f)) {
    return IFuncType::omp;
  }
  return IFuncType::standard;
}

}  // namespace detail

// enum class IFunc : unsigned {
//   heap,
//   stack,
//   global,
//   free,
//   scope,
//   heap_omp,
//   stack_omp,
//   free_omp,
//   scope_omp,
//   heap_cuda,
//   free_cuda,
// };

IFunc ifunc_for_function(IFunc general_type, llvm::Value* value) {
  detail::IFuncType type = typeart::detail::IFuncType::standard;

  if (auto function = llvm::dyn_cast<Function>(value)) {
    type = detail::ifunc_type_for(function);
  } else if (auto alloca = llvm::dyn_cast<AllocaInst>(value)) {
    type = detail::ifunc_type_for(alloca->getFunction());
  } else if (auto global = llvm::dyn_cast<GlobalVariable>(value)) {
    type = detail::ifunc_type_for(nullptr);
  } else if (auto callbase = llvm::dyn_cast<CallBase>(value)) {
    type            = detail::ifunc_type_for(callbase->getFunction());
    auto maybe_cuda = detail::ifunc_type_for(callbase->getCalledFunction());
    if (maybe_cuda == detail::IFuncType::cuda) {
      type = detail::IFuncType::cuda;
    }
  }

  if (detail::IFuncType::standard == type) {
    return general_type;
  }

  if (detail::IFuncType::cuda == type) {
    switch (general_type) {
      case IFunc::heap:
        return IFunc::heap_cuda;
      case IFunc::free:
        return IFunc::free_cuda;
      case IFunc::stack:
        return IFunc::stack;
      default:
        llvm_unreachable("IFunc not supported for CUDA.");
    }
  }

  switch (general_type) {
    case IFunc::stack:
      return IFunc::stack_omp;
    case IFunc::heap:
      return IFunc::heap_omp;
    case IFunc::free:
      return IFunc::free_omp;
    case IFunc::scope:
      return IFunc::scope_omp;
    default:
      llvm_unreachable("IFunc type is not supported for OpenMP.");
  }
}

TAFunctionDeclarator::TAFunctionDeclarator(Module& m, InstrumentationHelper&, TAFunctions& tafunc)
    : m(m), tafunc(tafunc) {
}

llvm::Function* TAFunctionDeclarator::make_function(IFunc id, llvm::StringRef basename,
                                                    llvm::ArrayRef<llvm::Type*> args) {
  const auto make_fname = [&id](llvm::StringRef name, llvm::ArrayRef<llvm::Type*> args) {
    std::string fname;
    llvm::raw_string_ostream os(fname);
    os << name;
    os << detail::get_func_suffix(id);
    //    if (!fixed_name) {
    //      os << "_" << std::to_string(args.size());
    //    }
    //    if (with_omp) {
    //      os << "_"
    //         << "omp";
    //    }
    return os.str();
  };

  const auto name = make_fname(basename, args);

  if (auto it = f_map.find(name); it != f_map.end()) {
    return it->second;
  }

  auto& c                           = m.getContext();
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
  const auto do_make = [&](auto& name, auto f_type) {
    const bool has_func_declared = m.getFunction(name) != nullptr;
    auto func_in_module          = m.getOrInsertFunction(name, f_type);

    Function* function{nullptr};
    if (has_func_declared) {
      LOG_WARNING("Function " << name << " is already declared in the module.")
      function = dyn_cast<Function>(func_in_module.getCallee()->stripPointerCasts());
    } else {
      function = dyn_cast<Function>(func_in_module.getCallee());
      setFunctionLinkageExternal(function);
    }

    addOptimizerAttributes(function);
    return function;
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
