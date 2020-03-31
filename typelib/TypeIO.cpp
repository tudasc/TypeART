//
// Created by sebastian on 22.03.18.
//

#include "TypeIO.h"
#include "TypeDB.h"
#include "TypeInterface.h"

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace llvm::yaml;

// template <>
// struct llvm::yaml::ScalarEnumerationTraits<typeart_type_kind_t> {
//  static void enumeration(IO& io, typeart_type_kind_t& value) {
//    io.enumCase(value, "builtin", BUILTIN);
//    io.enumCase(value, "struct", STRUCT);
//    io.enumCase(value, "pointer", POINTER);
//  }
//};
//
// template <>
// struct llvm::yaml::MappingTraits<typeart_type_info_t> {
//  static void mapping(IO& io, typeart_type_info_t& info) {
//    io.mapRequired("id", info.id);
//    io.mapRequired("kind", info.kind);
//  }
//};
//
// LLVM_YAML_IS_SEQUENCE_VECTOR(typeart_type_info_t)

template <>
struct llvm::yaml::MappingTraits<typeart::StructTypeInfo> {
  static void mapping(IO& io, typeart::StructTypeInfo& info) {
    io.mapRequired("id", info.id);
    io.mapRequired("name", info.name);
    io.mapRequired("extent", info.extent);
    io.mapRequired("member_count", info.numMembers);
    io.mapRequired("offsets", info.offsets);
    io.mapRequired("types", info.memberTypes);
    io.mapRequired("sizes", info.arraySizes);
    io.mapRequired("flags", info.flags);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(typeart::StructTypeInfo)

namespace typeart {

TypeIO::TypeIO(TypeDB& typeDB) : typeDB(typeDB) {
}

bool TypeIO::load(const std::string& file) {
  using namespace llvm;
  auto memBuffer = MemoryBuffer::getFile(file);

  if (std::error_code ec = memBuffer.getError()) {
    // TODO meaningful error handling/message
    return false;
  }

  typeDB.clear();

  yaml::Input in(memBuffer.get()->getMemBufferRef());
  std::vector<StructTypeInfo> structures;
  in >> structures;

  for (auto& typeInfo : structures) {
    typeDB.registerStruct(typeInfo);
  }

  if (in.error()) {
    // FIXME we really need meaningful errors for IO
    return false;
  }

  return true;
}

bool TypeIO::store(const std::string& file) const {
  using namespace llvm;

  std::error_code ec;
  raw_fd_ostream oss(StringRef(file), ec, sys::fs::OpenFlags::F_Text);

  if (oss.has_error()) {
    llvm::errs() << "Error\n";
    return false;
  }
  auto types = typeDB.getStructList();
  yaml::Output out(oss);
  if (types.size() > 0) {
    out << types;
  } else {
    out.beginDocuments();
    out.endDocuments();
  }

  return true;
}

}  // namespace typeart
