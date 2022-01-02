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

#include "InstrumentationHelper.h"

#include "support/Logger.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

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
