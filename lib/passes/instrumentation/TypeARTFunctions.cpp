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

#include "configuration/Configuration.h"
#include "support/ConfigurationBase.h"
#include "support/Logger.h"
#include "support/OmpUtil.h"

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

#include <memory>
#include <optional>
#include <string>

namespace typeart {
class InstrumentationHelper;
}  // namespace typeart

using namespace llvm;

namespace typeart {

namespace detail {
std::string get_func_suffix(IFunc id) {
  switch (id) {
    // case IFunc::free_cuda:
    // case IFunc::heap_cuda:
    //   return "_cuda";
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

  if (util::omp::isOmpContext(f)) {
    return IFuncType::omp;
  }

  return IFuncType::standard;
}

}  // namespace detail

IFunc ifunc_for_function(IFunc general_type, llvm::Value* value) {
  detail::IFuncType type = typeart::detail::IFuncType::standard;

  if (auto function = llvm::dyn_cast<Function>(value)) {
    type = detail::ifunc_type_for(function);
  } else if (auto alloca = llvm::dyn_cast<AllocaInst>(value)) {
    type = detail::ifunc_type_for(alloca->getFunction());
  } else if (llvm::isa<GlobalVariable>(value)) {
    type = detail::ifunc_type_for(nullptr);
  } else if (auto callbase = llvm::dyn_cast<CallBase>(value)) {
    type = detail::ifunc_type_for(callbase->getFunction());
    // auto maybe_cuda = detail::ifunc_type_for(callbase->getCalledFunction());
    // if (maybe_cuda == detail::IFuncType::cuda) {
    //   type = detail::IFuncType::cuda;
    // }
  }

  if (detail::IFuncType::standard == type) {
    return general_type;
  }

  // if (detail::IFuncType::cuda == type) {
  //   switch (general_type) {
  //     case IFunc::heap:
  //       return IFunc::heap_cuda;
  //     case IFunc::free:
  //       return IFunc::free_cuda;
  //     default:
  //       return general_type;
  //       //        llvm_unreachable("IFunc not supported for CUDA.");
  //   }
  // }

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

class TAFunctions final : public TAFunctionQuery {
  // densemap has problems with IFunc
  using FMap = std::unordered_map<IFunc, llvm::Function*>;
  FMap typeart_callbacks;
  FMap typeart_callbacks_alternatives;
  bool with_alternative_{false};

 public:
  explicit TAFunctions(bool with_alternatives) : with_alternative_(with_alternatives) {
  }
  llvm::Function* getFunctionFor(IFunc id) const override;
  void putFunctionFor(IFunc id, llvm::Function* f, bool alternative = false);
  void putAlternativeFunctionFor(IFunc id, llvm::Function* f);
};

class TAFunctionDeclarator {
  llvm::Module& module;
  //  [[maybe_unused]] InstrumentationHelper& instr;
  TAFunctions& typeart_functions;
  llvm::StringMap<llvm::Function*> function_map;

 public:
  TAFunctionDeclarator(llvm::Module& m, InstrumentationHelper& instr, TAFunctions& typeart_func);
  llvm::Function* make_function(IFunc function, llvm::StringRef basename, llvm::ArrayRef<llvm::Type*> args,
                                bool alternative = false);
  const llvm::StringMap<llvm::Function*>& getFunctionMap() const;
  virtual ~TAFunctionDeclarator() = default;
};

TAFunctionDeclarator::TAFunctionDeclarator(Module& mod, InstrumentationHelper&, TAFunctions& typeart_funcs)
    : module(mod), typeart_functions(typeart_funcs) {
}

llvm::Function* TAFunctionDeclarator::make_function(IFunc func_id, llvm::StringRef basename,
                                                    llvm::ArrayRef<llvm::Type*> args, bool alternative) {
  const auto make_fname = [&func_id](llvm::StringRef name, llvm::ArrayRef<llvm::Type*> callback_arguments) {
    std::string fname;
    llvm::raw_string_ostream os(fname);
    os << name;
    os << detail::get_func_suffix(func_id);

    // if (!fixed_name) {
    //   os << "_" << std::to_string(callback_arguments.size());
    // }
    // if (with_omp_postfix) {
    //   os << "_"
    //      << "omp";
    // }
    return os.str();
  };

  const auto name = make_fname(basename, args);

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
      LOG_DEBUG("Function " << function_name << " is already declared in the module.")
      function = dyn_cast<Function>(func_in_module.getCallee()->stripPointerCasts());
    } else {
      function = dyn_cast<Function>(func_in_module.getCallee());
      setFunctionLinkageExternal(function);
    }

    addOptimizerAttributes(function);
    return function;
  };

  auto* generated_function = do_make(name, FunctionType::get(Type::getVoidTy(c), args, false));

  function_map[name] = generated_function;

  if (alternative) {
    typeart_functions.putAlternativeFunctionFor(func_id, generated_function);
  } else {
    typeart_functions.putFunctionFor(func_id, generated_function);
  }
  return generated_function;
}

const llvm::StringMap<llvm::Function*>& TAFunctionDeclarator::getFunctionMap() const {
  return function_map;
}

Function* TAFunctions::getFunctionFor(IFunc id) const {
  const auto find_ = [&](const auto& map_) -> std::optional<Function*> {
    const auto element = map_.find(id);
    if (element == std::end(map_)) {
      LOG_WARNING("No functions for id " << int(id))
      return {};
    }
    return element->second;
  };

  if (with_alternative_) {
    auto result = find_(typeart_callbacks_alternatives);
    if (result) {
      return result.value();
    }
  }

  auto result = find_(typeart_callbacks);
  return result.value_or(nullptr);
}

void TAFunctions::putFunctionFor(IFunc id, llvm::Function* f, bool alternative) {
  if (alternative) {
    typeart_callbacks_alternatives[id] = f;
    return;
  }
  typeart_callbacks[id] = f;
}

namespace callbacks {

struct TypeArtFunc {
  const std::string name;
  llvm::Value* f{nullptr};
};

TypeArtFunc typeart_alloc{"__typeart_alloc"};
TypeArtFunc typeart_alloc_global{"__typeart_alloc_global"};
TypeArtFunc typeart_alloc_stack{"__typeart_alloc_stack"};
TypeArtFunc typeart_free{"__typeart_free"};
TypeArtFunc typeart_leave_scope{"__typeart_leave_scope"};

TypeArtFunc typeart_alloc_omp        = typeart_alloc;
TypeArtFunc typeart_alloc_stacks_omp = typeart_alloc_stack;
TypeArtFunc typeart_free_omp         = typeart_free;
TypeArtFunc typeart_leave_scope_omp  = typeart_leave_scope;

TypeArtFunc typeart_alloc_mty{"__typeart_alloc_mty"};
TypeArtFunc typeart_alloc_stack_mty{"__typeart_alloc_stack_mty"};
TypeArtFunc typeart_alloc_global_mty{"__typeart_alloc_global_mty"};
TypeArtFunc typeart_register_type{"__typeart_register_type"};

}  // namespace callbacks

std::unique_ptr<TAFunctionQuery> declare_instrumentation_functions(llvm::Module& m,
                                                                   const config::Configuration& configuration) {
  using namespace callbacks;
  auto functions = std::make_unique<TAFunctions>(false);
  InstrumentationHelper instrumentation_helper;
  instrumentation_helper.setModule(m);
  TAFunctionDeclarator decl(m, instrumentation_helper, *functions.get());

  auto alloc_arg_types      = instrumentation_helper.make_parameters(IType::ptr, IType::type_id, IType::extent);
  auto free_arg_types       = instrumentation_helper.make_parameters(IType::ptr);
  auto leavescope_arg_types = instrumentation_helper.make_parameters(IType::stack_count);

  const bool module_local_types = configuration[config::ConfigStdArgs::instrumentation];
  if (module_local_types) {
    auto alloc_arg_types_mty = instrumentation_helper.make_parameters(IType::ptr, IType::ptr, IType::extent);
    typeart_alloc.f          = decl.make_function(IFunc::heap, typeart_alloc_mty.name, alloc_arg_types_mty);
    typeart_register_type.f  = decl.make_function(IFunc::type, typeart_register_type.name, free_arg_types);
  } else {
    typeart_alloc.f = decl.make_function(IFunc::heap, typeart_alloc.name, alloc_arg_types);
  }

  typeart_alloc_stack.f  = decl.make_function(IFunc::stack, typeart_alloc_stack.name, alloc_arg_types);
  typeart_alloc_global.f = decl.make_function(IFunc::global, typeart_alloc_global.name, alloc_arg_types);
  typeart_free.f         = decl.make_function(IFunc::free, typeart_free.name, free_arg_types);
  typeart_leave_scope.f  = decl.make_function(IFunc::scope, typeart_leave_scope.name, leavescope_arg_types);

  typeart_alloc_omp.f        = decl.make_function(IFunc::heap_omp, typeart_alloc_omp.name, alloc_arg_types);
  typeart_alloc_stacks_omp.f = decl.make_function(IFunc::stack_omp, typeart_alloc_stacks_omp.name, alloc_arg_types);
  typeart_free_omp.f         = decl.make_function(IFunc::free_omp, typeart_free_omp.name, free_arg_types);

  typeart_leave_scope_omp.f = decl.make_function(IFunc::scope_omp, typeart_leave_scope_omp.name, leavescope_arg_types);

  return functions;
}

}  // namespace typeart
