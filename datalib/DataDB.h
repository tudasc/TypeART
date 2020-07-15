//
// Created by ahueck on 14.07.20.
//

#ifndef TYPEART_DATADB_H
#define TYPEART_DATADB_H
#include "TaData.h"
namespace typeart {
class DataDB {
  data::ModuleDataVec modules;

 public:
  void clear();
  void clearEmpty();
  void makeUnique() {
  }
  void putModule(data::ModuleData& data);
  const data::ModuleDataVec& getModules();

  data::ModuleData& module(data::MID id);
  data::FunctionData& function(data::MID id, data::FID fid);
};
}  // namespace typeart
#endif  // TYPEART_DATADB_H
