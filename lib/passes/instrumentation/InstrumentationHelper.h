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

#ifndef LIB_INSTRUMENTATIONHELPER_H_
#define LIB_INSTRUMENTATIONHELPER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"

#include <cstddef>
#include <functional>
#include <map>
#include <string>

namespace llvm {
class Function;
class Module;
class Value;
class Type;
class ConstantInt;
}  // namespace llvm

namespace typeart {

enum class IType {
  ptr,          // Type for passing a pointer to the runtime
  function_id,  // Type for identifying a function
  type_id,      // Type for identifying a type
  extent,       // Type for identifying an array length
  alloca_id,    // Type for identifying a memory allocation
  stack_count,  // Type for identifying a count of stack alloca instructions
};

class InstrumentationHelper {
  llvm::Module* module{nullptr};

 public:
  InstrumentationHelper();
  void setModule(llvm::Module& m);
  llvm::Module* getModule() const;
  static llvm::SmallVector<llvm::Type*, 8> make_signature(const llvm::ArrayRef<llvm::Value*>& args);

  template <typename... Types>
  llvm::SmallVector<llvm::Type*, 8> make_parameters(Types... args) {
    static_assert((std::is_same_v<IType, Types> && ...));
    return {getTypeFor(args)...};
  }

  llvm::Type* getTypeFor(IType id);
  llvm::ConstantInt* getConstantFor(IType id, size_t val = 0);
  const std::map<std::string, llvm::Function*>& getFunctionMap() const;

  virtual ~InstrumentationHelper();
};

}  // namespace typeart

#endif /* LIB_INSTRUMENTATIONHELPER_H_ */
