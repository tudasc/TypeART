//
// Created by ahueck on 08.10.20.
//

#include "TypeARTFunctions.h"

#include "llvm/IR/Argument.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;

namespace typeart {
TAFunctionDeclarator::TAFunctionDeclarator(Module& m, InstrumentationHelper& instr, TAFunctions& tafunc)
    : m(m), instr(instr), tafunc(tafunc) {
}

llvm::Function* TAFunctionDeclarator::make_function(IFunc id, llvm::StringRef basename,
                                                    llvm::ArrayRef<llvm::Type*> args, bool fixed_name) {
  const auto make_fname = [&fixed_name](llvm::StringRef name, llvm::ArrayRef<llvm::Type*> args) {
    if (fixed_name) {
      return std::string(name.str());
    }
    return std::string((name + "_" + std::to_string(args.size())).str());
  };

  const auto name = make_fname(basename, args);
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
    auto fc = m.getOrInsertFunction(name, f_type);
#if LLVM_VERSION >= 10
    auto f = dyn_cast<Function>(fc.getCallee());
#else
    auto f = dyn_cast<Function>(fc);
#endif
    setFunctionLinkageExternal(f);
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
