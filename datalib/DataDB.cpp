//
// Created by ahueck on 14.07.20.
//

#include "DataDB.h"

#include <algorithm>

using namespace typeart::data;
namespace typeart {

void DataDB::clear() {
}

void DataDB::putModule(data::ModuleData& moduleData) {
  modules.push_back(moduleData);
}

const data::ModuleDataVec& DataDB::getModules() {
  return modules;
}
data::ModuleData& DataDB::module(data::MID id) {
  auto it = std::find_if(std::begin(modules), std::end(modules), [&id](ModuleData& m) { return id == m.id; });
  if (it != std::end(modules)) {
    return *it;
  }
  return modules.emplace_back(id);
}

data::FunctionData& DataDB::function(data::MID id, data::FID fid) {
  auto& mdata          = module(id);
  auto& functions      = mdata.functions;
  auto&& [it, success] = functions.try_emplace(fid, FunctionData{fid});

  return it->second;
}

}  // namespace typeart