//
// Created by ahueck on 15.07.20.
//

#include "ModuleManagerBase.h"

#include "DataIO.h"
#include "Logger.h"
#include "Util.h"

#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>

using namespace typeart::data;
namespace typeart {

ModuleManagerBase::ModuleManagerBase(std::string path) : path(std::move(path)) {
}

FID ModuleManagerBase::lookupFunction(llvm::Function& f) {
  const auto fname = f.getName();

  if (auto it = f_map.find(fname); it != f_map.end()) {
    auto id = it->second;
    c.f     = id;
    return id;
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

  c.f = fID;
  return fID;
}

MID ModuleManagerBase::lookupModule(llvm::Module& m) {
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

void ModuleManagerBase::clearEmpty() {
  mDB.clearEmpty();
}

void ModuleManagerBase::make_id(AllocData& d) {
  // TODO duplicate checking etc.?
  ++aId;
  d.id = aId;
}

AllocData ModuleManagerBase::make_data(int type, llvm::Instruction& i) {
  AllocData data;
  make_id(data);
  data.typeID = type;
  data.dump   = util::dump(i);
  auto dbg    = util::getDebugVar(i);
  if (dbg != nullptr) {
    data.dbg.line = dbg->getLine();
  }
  return data;
}

bool ModuleManagerBase::load() {
  const auto max_count = [](const auto& vec, auto& max) {
    auto max_e = std::max_element(std::begin(vec), std::end(vec),
                                  [](const auto& item, const auto& item2) { return item.first < item2.first; });
    if (max_e != std::end(vec) && max_e->first > max) {
      max = max_e->first;
    }
  };

  DataIO io(mDB);
  if (io.load(path)) {
    mDB.makeUnique();

    const auto& modules = mDB.getModules();
    for (const auto& m : modules) {
      if (m.id > mID) {
        mID = m.id;
      }
      m_map.try_emplace(m.name, m.id);
      max_count(m.globals, aId);
      max_count(m.functions, fID);
      for (auto&& [id, f] : m.functions) {
        f_map.try_emplace(f.name, f.id);
        max_count(f.heap, aId);
        max_count(f.stack, aId);
      }
    }
    return true;
  }
  return false;
}

bool ModuleManagerBase::store() {
  DataIO io(mDB);
  return io.store(path);
}
}  // namespace typeart