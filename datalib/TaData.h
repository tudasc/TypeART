//
// Created by ahueck on 14.07.20.
//

#ifndef TYPEART_TADATA_H
#define TYPEART_TADATA_H

#include <string>
#include <unordered_map>
#include <vector>

namespace typeart {
namespace data {
using MID      = unsigned;
using FID      = unsigned;
using AllocID  = unsigned;
using Location = int;

struct FilterData final {
  std::string reason{"-"};
};

struct DbgData final {
  Location line{-1};
  std::string name{""};
};

struct AllocData final {
  AllocID id{0};
  int typeID{0};
  std::string typeStr{""};
  std::string dump{""};
  DbgData dbg{};
  FilterData filter{};
};

struct FreeData final {
  std::string dump{""};
  DbgData dbg{};
};

using AllocDataMap = std::unordered_map<AllocID, AllocData>;

struct FunctionData final {
  FunctionData() {
  }
  FunctionData(FID id) : id(id) {
  }
  FID id{0};
  std::string name{""};
  std::string demangled{""};
  AllocDataMap stack{};
  AllocDataMap heap{};
  std::vector<FreeData> frees{};
};

using FunctionDataMap = std::unordered_map<FID, FunctionData>;

struct ModuleData final {
  ModuleData() {
  }
  ModuleData(MID id) : id(id) {
  }
  MID id{0};
  std::string name{""};
  AllocDataMap globals{};
  FunctionDataMap functions{};
  int globs{}, heap{}, stack{};
};

using ModuleDataVec = std::vector<ModuleData>;

}  // namespace data
}  // namespace typeart

#endif  // TYPEART_TADATA_H