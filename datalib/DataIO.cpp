//
// Created by ahueck on 14.07.20.
//

#include "DataIO.h"

#include "DataDB.h"
#include "TaData.h"

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace llvm::yaml;
using namespace llvm;
using namespace typeart;
using namespace typeart::data;

namespace detail {
template <typename T>
struct map2vec {
  using VecTy = T;
  template <typename Key, typename Val>
  static typename VecTy::value_type construct(Key&& key, Val&& val) {
    return {std::forward<Val>(val)};
  }
};

template <typename Map>
struct vec2map {
  using MapTy = Map;
  template <typename Val>
  static typename MapTy::value_type construct(Val&& val) {
    return {val.id, val};
  }
};

template <typename Map, typename Vec>  //, typename VecPushF, typename MapPushF
struct YamlNormalizeMap {
  using VecTy = Vec;
  using MapTy = Map;
  //   using value_type = typename Vec::value_type;

  VecTy norm;

  YamlNormalizeMap(IO& io) : norm() {
  }

  YamlNormalizeMap(IO&, MapTy& polar) {
    for (auto&& [key, val] : polar) {
      norm.emplace_back(map2vec<VecTy>::construct(key, val));
    }
  }

  MapTy denormalize(IO&) {
    MapTy m;
    for (const auto& val : norm) {
      m.emplace(vec2map<MapTy>::construct(val));
    }
    return m;
  }
};

}  // namespace detail

using NormalizedAMap = ::detail::YamlNormalizeMap<typeart::data::AllocDataMap, std::vector<typeart::data::AllocData>>;
using NormalizeAllocData = MappingNormalization<NormalizedAMap, NormalizedAMap::MapTy>;

using NormalizedFMap =
    ::detail::YamlNormalizeMap<typeart::data::FunctionDataMap, std::vector<typeart::data::FunctionData>>;
using NormalizeFuncData = MappingNormalization<NormalizedFMap, NormalizedFMap::MapTy>;

template <>
struct llvm::yaml::MappingTraits<typeart::data::FilterData> {
  static const bool flow = true;
  static void mapping(IO& io, typeart::data::FilterData& info) {
    io.mapRequired("reason", info.reason);
  }
};

template <>
struct llvm::yaml::MappingTraits<typeart::data::DbgData> {
  static const bool flow = true;
  static void mapping(IO& io, typeart::data::DbgData& info) {
    io.mapRequired("name", info.name);
    io.mapRequired("loc", info.line);
  }
};

template <>
struct llvm::yaml::MappingTraits<typeart::data::AllocData> {
  static const bool flow = true;
  static void mapping(IO& io, typeart::data::AllocData& info) {
    io.mapRequired("id", info.id);
    io.mapRequired("ty", info.typeID);
    io.mapOptional("ty_n", info.typeStr);
    io.mapRequired("dump", info.dump);
    io.mapOptional("dbg", info.dbg);
    io.mapOptional("filter", info.filter);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(AllocData)

/*
template <>
struct llvm::yaml::MappingTraits<typeart::data::AllocDataMap> {
  static void mapping(IO& io, typeart::data::AllocDataMap& polar) {
    using NormalizedMap = ::detail::YamlNormalizeMap<AllocDataMap, std::vector<AllocData>>;
    MappingNormalization<NormalizedMap, NormalizedMap::MapTy> keys(io, polar);
    io.mapRequired("sequence", keys->norm);
  }
};
*/

template <>
struct llvm::yaml::MappingTraits<typeart::data::FunctionData> {
  static void mapping(IO& io, typeart::data::FunctionData& info) {
    io.mapRequired("id", info.id);
    io.mapRequired("name", info.name);

    NormalizeAllocData keys_h(io, info.heap);
    NormalizeAllocData keys_s(io, info.stack);
    io.mapOptional("heap", keys_h->norm);
    io.mapOptional("stack", keys_s->norm);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionData)

/*
template <>
struct llvm::yaml::MappingTraits<typeart::data::FunctionDataMap> {
  static void mapping(IO& io, typeart::data::FunctionDataMap& polar) {
    using NormalizedMap = ::detail::YamlNormalizeMap<FunctionDataMap, std::vector<FunctionData>>;
    MappingNormalization<NormalizedMap, NormalizedMap::MapTy> keys(io, polar);
    io.mapRequired("sequence", keys->norm);
  }
};
*/

template <>
struct llvm::yaml::MappingTraits<typeart::data::ModuleData> {
  static void mapping(IO& io, typeart::data::ModuleData& info) {
    io.mapRequired("id", info.id);
    io.mapRequired("name", info.name);

    NormalizeFuncData keys_f(io, info.functions);
    NormalizeAllocData keys_g(io, info.globals);

    io.mapOptional("globals", keys_g->norm);
    io.mapOptional("functions", keys_f->norm);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(ModuleData)

namespace typeart {

DataIO::DataIO(DataDB& dataDB) : dataDB(dataDB) {
}

bool DataIO::load(const std::string& file) {
  using namespace llvm;
  auto memBuffer = MemoryBuffer::getFile(file);

  if (std::error_code ec = memBuffer.getError()) {
    // TODO meaningful error handling/message
    return false;
  }

  dataDB.clear();

  yaml::Input in(memBuffer.get()->getMemBufferRef());
  std::vector<ModuleData> modules;
  in >> modules;

  for (auto& moduleInfo : modules) {
    dataDB.putModule(moduleInfo);
  }

  if (in.error()) {
    // FIXME we really need meaningful errors for IO
    return false;
  }

  return true;
}

bool DataIO::store(const std::string& file) const {
  using namespace llvm;

  std::error_code ec;
  raw_fd_ostream oss(StringRef(file), ec, sys::fs::OpenFlags::F_Text);

  if (oss.has_error()) {
    llvm::errs() << "Error\n";
    return false;
  }
  auto modules = dataDB.getModules();
  yaml::Output out(oss);
  if (modules.size() > 0) {
    out << modules;
  } else {
    out.beginDocuments();
    out.endDocuments();
  }

  return true;
}

}  // namespace typeart