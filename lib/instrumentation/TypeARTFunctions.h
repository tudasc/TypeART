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

// TypeArtFunc typeart_alloc{"__typeart_alloc"};
// TypeArtFunc typeart_alloc_global{"__typeart_alloc_global"};
// TypeArtFunc typeart_alloc_stack{"__typeart_alloc_stack"};
// TypeArtFunc typeart_free{"__typeart_free"};
// TypeArtFunc typeart_leave_scope{"__typeart_leave_scope"};

enum class IFunc : unsigned {
  heap,
  stack,
  global,
  free,
  scope,
};

class TAFunctionQuery {
 public:
  virtual llvm::Function* getFunctionFor(IFunc id) = 0;
  virtual ~TAFunctionQuery()                       = default;
};

class TAFunctions : public TAFunctionQuery {
  // densemap has problems with IFunc
  using FMap = std::unordered_map<IFunc, llvm::Function*>;
  FMap typeart_callbacks;

 public:
  TAFunctions();

  llvm::Function* getFunctionFor(IFunc id) override;
  void putFunctionFor(IFunc id, llvm::Function* f);

  virtual ~TAFunctions() = default;
};

class TAFunctionDeclarator {
  llvm::Module& m;
  InstrumentationHelper& instr;
  TAFunctions& tafunc;
  llvm::StringMap<llvm::Function*> f_map;

 public:
  TAFunctionDeclarator(llvm::Module& m, InstrumentationHelper& instr, TAFunctions& tafunc);
  llvm::Function* make_function(IFunc id, llvm::StringRef basename, llvm::ArrayRef<llvm::Type*> args,
                                bool fixed_name = true);
  const llvm::StringMap<llvm::Function*>& getFunctionMap() const;
  virtual ~TAFunctionDeclarator() = default;
};

}  // namespace typeart

#endif  // TYPEART_TYPEARTFUNCTIONS_H
