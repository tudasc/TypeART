//
// Created by ahueck on 14.07.20.
//

#ifndef TYPEART_MODULEDATAMANAGER_H
#define TYPEART_MODULEDATAMANAGER_H

#include "ModuleManagerBase.h"
#include "analysis/MemOpVisitor.h"

namespace llvm {
class Function;
class Module;
class Value;
class GlobalVariable;
class Instruction;
class CallInst;
}  // namespace llvm

using namespace typeart::data;

namespace typeart {

class TypeManager;

class ModuleDataManager : public ModuleManagerBase {
  TypeManager* type_m{nullptr};

 protected:
  AllocData make_data(int, llvm::Instruction&) override;

 public:
  explicit ModuleDataManager(std::string path);

  void setTypeManager(TypeManager* m);

  void putFree(const llvm::CallInst&, std::string filter = {""});
  data::AllocID putHeap(const MallocData&, int type, std::string filter = {""});
  data::AllocID putStack(const AllocaData&, int type, std::string filter = {""});
  data::AllocID putGlobal(llvm::GlobalVariable*, int type, std::string filter = {""});

  virtual ~ModuleDataManager() = default;
};

}  // namespace typeart

#endif  // TYPEART_MODULEDATAMANAGER_H
