#ifndef _LIB_MUSTSUPPORTPASS_H
#define _LIB_MUSTSUPPORTPASS_H

#include "TypeDB.h"
#include "TypeManager.h"

#include "llvm/Pass.h"

#include <string>

namespace llvm {
class Constant;
class Module;
class Function;
class AnalysisUsage;
class Instruction;
}  // namespace llvm

namespace typeart {
namespace pass {

class TypeArtPass : public llvm::ModulePass {
 private:
  struct TypeArtFunc {
    const std::string name{""};
    llvm::Constant* f{nullptr};
  };

  TypeArtFunc typeart_alloc{"__typeart_alloc"};
  TypeArtFunc typeart_alloc_global{"__typeart_alloc_global"};
  TypeArtFunc typeart_alloc_stack{"__typeart_alloc_stack"};
  TypeArtFunc typeart_free{"__typeart_free"};
  TypeArtFunc typeart_leave_scope{"__typeart_leave_scope"};
  TypeArtFunc typeart_assert_type{"__typeart_assert_type"};
  TypeArtFunc typeart_assert_type_len{"__typeart_assert_type_len"};
  TypeArtFunc typeart_assert_tycart{"__tycart_assert"};
  TypeArtFunc typeart_assert_tycart_auto{"__tycart_assert_auto"};
  TypeArtFunc typeart_assert_tycart_fti_t{"__tycart_register_FTI_t"};

  TypeManager typeManager;

  // Call/Invoke Fix
  template <typename T, typename U>
  struct Wrap {
    union {
      T* c;
      U* i;
    };

    short active;

    llvm::Value* getArgOperand(int pos) {
      switch (active) {
        case 1:
          return c->getArgOperand(pos);
        case 2:
          return i->getArgOperand(pos);
        default:
          assert(false);
      }
    }

    llvm::Instruction* inst() {
      return c;
    }
  };

 public:
  static char ID;  // used to identify pass

  TypeArtPass();
  bool doInitialization(llvm::Module&) override;
  bool runOnModule(llvm::Module&) override;
  bool runOnFunc(llvm::Function&);
  bool doFinalization(llvm::Module&) override;
  void getAnalysisUsage(llvm::AnalysisUsage&) const override;

 private:
  void declareInstrumentationFunctions(llvm::Module&);
  void propagateTypeInformation(llvm::Module&);
  void printStats(llvm::raw_ostream&);
};

}  // namespace pass
}  // namespace typeart

#endif
