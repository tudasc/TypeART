#include "InstrumentationHelper.h"

#include "support/Logger.h"

#include "llvm/IR/Argument.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

namespace typeart {

using namespace llvm;

InstrumentationHelper::InstrumentationHelper() = default;
InstrumentationHelper::~InstrumentationHelper() = default;

llvm::SmallVector<llvm::Type*, 8> InstrumentationHelper::make_signature(llvm::LLVMContext& /*c*/,
                                                                        const llvm::ArrayRef<llvm::Value*>& args) {
  llvm::SmallVector<llvm::Type*, 8> types;
  for (auto* val : args) {
    types.push_back(val->getType());
  }
  return types;
}

llvm::Type* InstrumentationHelper::getTypeFor(IType id) {
  auto& c = module->getContext();
  switch (id) {
    case IType::ptr:
      return Type::getInt8PtrTy(c);
    case IType::function_id:
      return Type::getInt32Ty(c);
    case IType::extent:
      return Type::getInt64Ty(c);
    case IType::type_id:
      [[fallthrough]];
    case IType::alloca_id:
      return Type::getInt32Ty(c);
    case IType::stack_count:
      return Type::getInt32Ty(c);
    default:
      return nullptr;
  }
}

llvm::ConstantInt* InstrumentationHelper::getConstantFor(IType id, size_t value) {
  const auto make_int = [&]() -> Optional<ConstantInt*> {
    auto itype = dyn_cast_or_null<IntegerType>(getTypeFor(id));
    if (itype == nullptr) {
      LOG_FATAL("Pointer for the constant type is null, need aborting...");
      return None;
    }
    return ConstantInt::get(itype, value);
  };
  return make_int().getValue();
}

void InstrumentationHelper::setModule(llvm::Module& m) {
  module = &m;
}

llvm::Function* InstrumentationHelper::make_function(llvm::StringRef basename, llvm::ArrayRef<llvm::Type*> args,
                                                     bool fixed_name) {
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

  auto& m = *module;
  auto& c = m.getContext();
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

  return f;
}

const std::map<std::string, llvm::Function*>& InstrumentationHelper::getFunctionMap() const {
  return f_map;
}

}  // namespace typeart
