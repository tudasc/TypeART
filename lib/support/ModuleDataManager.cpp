//
// Created by ahueck on 14.07.20.
//

#include "ModuleDataManager.h"

#include "../analysis/MemOpVisitor.h"
#include "DataIO.h"
#include "Logger.h"
#include "Util.h"

#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>

using namespace typeart::data;
namespace typeart {

ModuleDataManager::ModuleDataManager(std::string path) : path(std::move(path)) {
}

FID ModuleDataManager::lookupFunction(llvm::Function& f) {
  const auto fname = f.getName();

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

  if (auto it = m_map.find(name); it != m_map.end()) {
    auto id = it->second;
    c.m     = id;
    return id;
  }

  // new module:
  ++mID;
  auto& mdata = mDB.module(mID);
  mdata.name  = name;

  if (auto&& [iter, success] = m_map.try_emplace(name, mID); !success) {
    LOG_ERROR("Error emplacing module data into functionMap.");
    return 0;
  }

  c.m = mID;
  return mID;
}

void ModuleDataManager::setContext(MID id) {
  c.m = id;
}

void ModuleDataManager::make_id(AllocData& d) {
  // TODO duplicate checking etc.?
  ++aId;
  d.id = aId;
}

AllocData ModuleDataManager::make_data(int type, llvm::Instruction& i) {
  AllocData data;
  make_id(data);
  data.typeID = type;
  auto dbg    = util::getDebugVar(i);
  return data;
}

data::AllocID ModuleDataManager::putHeap(FID fID, const MallocData& m, int type) {
  auto mid     = c.m;
  auto& fdata  = mDB.function(mid, fID);
  auto& heap_m = fdata.heap;

  auto data = make_data(type, *m.call);

  heap_m.try_emplace(data.id, data);
  return data.id;
}
data::AllocID ModuleDataManager::putStack(FID, const AllocaData& m, int type) {
  auto mid      = c.m;
  auto& fdata   = mDB.function(mid, fID);
  auto& stack_m = fdata.stack;

  auto data = make_data(type, *m.alloca);
  stack_m.try_emplace(data.id, data);
  return data.id;
}
data::AllocID ModuleDataManager::putGlobal(llvm::GlobalVariable* g, int type) {
  auto mid       = c.m;
  auto& mdata    = mDB.module(mid);
  auto& global_m = mdata.globals;

  AllocData data;
  make_id(data);
  data.typeID = type;
  // auto dbg    = util::getDebugVar(*g);

  global_m.try_emplace(data.id, data);
  return data.id;
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