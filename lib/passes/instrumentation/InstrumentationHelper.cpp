#include "InstrumentationHelper.h"

#include "support/Logger.h"

#include "llvm/IR/Argument.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

namespace typeart {

using namespace llvm;

InstrumentationHelper::InstrumentationHelper()  = default;
InstrumentationHelper::~InstrumentationHelper() = default;

llvm::SmallVector<llvm::Type*, 8> InstrumentationHelper::make_signature(const llvm::ArrayRef<llvm::Value*>& args) {
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
    // TyCart - BEGIN
    case IType::cp_id:
      return Type::getInt32Ty(c);
    case IType::type_size:
      return IType::getInt64Ty(c);
    // TyCart - END
    default:
      LOG_WARNING("Unknown IType selected.");
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

llvm::Module* InstrumentationHelper::getModule() const {
  return module;
}

}  // namespace typeart
