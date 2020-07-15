//
// Created by ahueck on 15.07.20.
//

#ifndef TYPEART_MODULEMANAGERBASE_H
#define TYPEART_MODULEMANAGERBASE_H

#include "../../datalib/DataDB.h"
#include "../../datalib/TaData.h"

#include <llvm/ADT/StringMap.h>
#include <string>

using namespace typeart::data;

namespace llvm {
class Function;
class Module;
class Instruction;
}  // namespace llvm

namespace typeart {

class ModuleManagerBase {
 protected:
  std::string path;
  DataDB mDB;
  MID mID{127};
  FID fID{127};
  AllocID aId{0};
  llvm::StringMap<FID> f_map;
  llvm::StringMap<MID> m_map;

  // TODO eventually caches current context :
  struct context {
    MID m;
    FID f;
  } c;

  void make_id(AllocData&);
  virtual AllocData make_data(int, llvm::Instruction&);

 public:
  explicit ModuleManagerBase(std::string path);
  FID lookupFunction(llvm::Function& f);
  MID lookupModule(llvm::Module& m);
  void clearEmpty();
  bool load();
  bool store();

  virtual ~ModuleManagerBase() = default;
};
}  // namespace typeart

#endif  // TYPEART_MODULEMANAGERBASE_H
