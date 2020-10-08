//
// Created by ahueck on 08.10.20.
//

#ifndef TYPEART_TYPEARTFUNCTIONS_H
#define TYPEART_TYPEARTFUNCTIONS_H
#include "InstrumentationHelper.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Function;
class Type;
class Module;
}  // namespace llvm

namespace typeart {

class InstrumentationHelper;

class TAFunctionDeclarator {
  llvm::Module& m;
  InstrumentationHelper& instr;
  llvm::StringMap<llvm::Function*> f_map;

 public:
  TAFunctionDeclarator(llvm::Module& m, InstrumentationHelper& instr);
  llvm::Function* make_function(llvm::StringRef basename, llvm::ArrayRef<llvm::Type*> args, bool fixed_name = true);
  const llvm::StringMap<llvm::Function*>& getFunctionMap() const;
  virtual ~TAFunctionDeclarator() = default;
};

}  // namespace typeart

#endif  // TYPEART_TYPEARTFUNCTIONS_H
