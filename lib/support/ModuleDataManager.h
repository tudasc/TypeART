//
// Created by ahueck on 14.07.20.
//

#ifndef TYPEART_MODULEDATAMANAGER_H
#define TYPEART_MODULEDATAMANAGER_H

//#include "../../datalib/TaData.h"

#include <DataDB.h>
#include <TaData.h>
#include <bits/unordered_map.h>
#include <llvm/ADT/StringMap.h>

namespace llvm {
class Function;
class Module;
class Value;
}  // namespace llvm

using namespace typeart::data;

namespace typeart {

class ModuleDataManager {
  std::string path;
  DataDB mDB;
  MID mID{127};
  FID fID{127};
  AllocID id{0};

  llvm::StringMap<FID> f_map;
  llvm::StringMap<MID> m_map;

  // TODO *unused* eventually caches current context :
  struct context {
    llvm::Function* cur_f;
    llvm::Module* cur_m;
    MID m;
    FID f;
    typeart::data::ModuleData* md;
    typeart::data::FunctionData* fd;
  } ctx;

 public:
  explicit ModuleDataManager(std::string path);
  FID lookupFunction(llvm::Function& f);
  MID lookupModule(llvm::Module& m);

  void putHeap(FID, llvm::Value*);
  void putStack(FID, llvm::Value*);
  void putGlobal(MID, llvm::Value*);

  bool load();
  bool store();
  virtual ~ModuleDataManager() = default;
};

}  // namespace typeart

#endif  // TYPEART_MODULEDATAMANAGER_H
