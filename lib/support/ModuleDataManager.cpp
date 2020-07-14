//
// Created by ahueck on 14.07.20.
//

#include "ModuleDataManager.h"

#include "../../datalib/TaData.h"
#include "Logger.h"
#include "Util.h"

#include <DataIO.h>
#include <TaData.h>
#include <llvm/IR/Module.h>

using namespace typeart::data;
namespace typeart {

ModuleDataManager::ModuleDataManager(std::string path) : path(std::move(path)) {
}

FID ModuleDataManager::lookupFunction(llvm::Function& f) {
  const auto fname = f.getName();
  if (ctx.cur_f != &f) {
    ctx.cur_f = &f;
  }

  if (auto it = f_map.find(fname); it != f_map.end()) {
    return it->second;
  }

  // new function
  ++fID;
  auto mid    = lookupModule(*f.getParent());
  auto& fdata = mDB.function(mid, fID);
  fdata.name  = fname;

  if (auto&& [iter, success] = f_map.try_emplace(fname, fID); !success) {
    LOG_ERROR("Error emplacing function data into functionMap.");
    return 0;
  }

  return fID;
}

MID ModuleDataManager::lookupModule(llvm::Module& m) {
  const std::string name = m.getName();
  if (ctx.cur_m != &m) {
    ctx.cur_m = &m;
  }
  if (auto it = m_map.find(name); it != m_map.end()) {
    return it->second;
  }

  // new module:
  ++mID;
  auto& mdata = mDB.module(mID);
  mdata.name  = name;

  if (auto&& [iter, success] = m_map.try_emplace(name, mID); !success) {
    LOG_ERROR("Error emplacing module data into functionMap.");
    return 0;
  }

  return mID;
}

bool ModuleDataManager::load() {
  DataIO io(mDB);
  if (io.load(path)) {
    mDB.makeUnique();
    // TODO set mID, fID, AllocID start values
  }
  return false;
}

bool ModuleDataManager::store() {
  DataIO io(mDB);
  return io.store(path);
}
}  // namespace typeart