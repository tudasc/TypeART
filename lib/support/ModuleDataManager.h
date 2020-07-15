//
// Created by ahueck on 14.07.20.
//

#ifndef TYPEART_MODULEDATAMANAGER_H
#define TYPEART_MODULEDATAMANAGER_H

#include "../../datalib/DataDB.h"
#include "../../datalib/TaData.h"
//#include "TaData.h"
#include "analysis/MemOpVisitor.h"

#include <bits/unordered_map.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>

namespace llvm {
class Function;
class Module;
class Value;
class GlobalVariable;
class Instruction;
}  // namespace llvm

using namespace typeart::data;

namespace typeart {

class ModuleDataManager {
  std::string path;
  DataDB mDB;
  MID mID{127};
  FID fID{127};
  AllocID aId{0};

  llvm::StringMap<FID> f_map;
  // llvm::DenseMap<FID, MID> f2m_map;
  llvm::StringMap<MID> m_map;

  // TODO eventually caches current context :
  struct context {
    MID m;
  } c;

 private:
  void make_id(AllocData&);
  AllocData make_data(int, llvm::Instruction&);
  // void make_alloc(AllocData&);

 public:
  explicit ModuleDataManager(std::string path);
  FID lookupFunction(llvm::Function& f);
  MID lookupModule(llvm::Module& m);

  void setContext(MID);
  data::AllocID putHeap(FID, const MallocData&, int type);
  data::AllocID putStack(FID, const AllocaData&, int type);
  data::AllocID putGlobal(llvm::GlobalVariable*, int type);

  bool load();
  bool store();
  virtual ~ModuleDataManager() = default;
};

}  // namespace typeart

#endif  // TYPEART_MODULEDATAMANAGER_H
