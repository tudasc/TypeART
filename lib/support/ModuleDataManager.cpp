//
// Created by ahueck on 14.07.20.
//

#include "ModuleDataManager.h"

#include "../TypeManager.h"
#include "../analysis/MemOpVisitor.h"
#include "DataIO.h"
#include "Logger.h"
#include "Util.h"

#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>

using namespace typeart::data;
namespace typeart {

ModuleDataManager::ModuleDataManager(std::string path) : ModuleManagerBase(std::move(path)) {
}

void ModuleDataManager::setTypeManager(TypeManager* m) {
  type_m = m;
}

AllocData ModuleDataManager::make_data(int type, llvm::Instruction& i) {
  auto data = ModuleManagerBase::make_data(type, i);
  if (type_m != nullptr) {
    data.typeStr = type_m->typeNameOf(type);
  }
  return data;
}

data::AllocID ModuleDataManager::putHeap(const MallocData& m, int type, std::string filter) {
  auto mid     = c.m;
  auto& fdata  = mDB.function(mid, c.f);
  auto& heap_m = fdata.heap;

  auto data          = make_data(type, *m.call);
  data.filter.reason = filter;

  heap_m.try_emplace(data.id, data);

  auto& mdata = mDB.module(mid);
  ++mdata.heap;
  return data.id;
}
data::AllocID ModuleDataManager::putStack(const AllocaData& m, int type, std::string filter) {
  auto mid      = c.m;
  auto& fdata   = mDB.function(mid, c.f);
  auto& stack_m = fdata.stack;

  auto data          = make_data(type, *m.alloca);
  data.filter.reason = filter;
  stack_m.try_emplace(data.id, data);

  auto& mdata = mDB.module(mid);
  ++mdata.stack;
  return data.id;
}
data::AllocID ModuleDataManager::putGlobal(llvm::GlobalVariable* g, int type, std::string filter) {
  auto mid       = c.m;
  auto& mdata    = mDB.module(mid);
  auto& global_m = mdata.globals;

  AllocData data;
  make_id(data);
  data.typeID        = type;
  data.dump          = util::dump(*g);
  data.filter.reason = filter;
  // auto dbg    = util::getDebugVar(*g);

  global_m.try_emplace(data.id, data);
  ++mdata.globs;
  return data.id;
}

}  // namespace typeart