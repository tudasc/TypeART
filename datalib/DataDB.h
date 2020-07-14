//
// Created by ahueck on 14.07.20.
//

#ifndef TYPEART_DATADB_H
#define TYPEART_DATADB_H
#include "TaData.h"
namespace typeart {
class DataDB {
 public:
  void clear();
  void putModule(data::ModuleData& data);
  data::ModuleDataVec getModules();
};
}  // namespace typeart
#endif  // TYPEART_DATADB_H
