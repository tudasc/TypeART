//
// Created by sebastian on 22.03.18.
//

#include "TypeIO.h"

#include "TypeDB.h"
#include "TypeInterface.h"

#include <algorithm>
#include <assert.h>

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/YAMLParser.h>
#include <llvm/Support/YAMLTraits.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm::yaml;

template <>
struct llvm::yaml::ScalarEnumerationTraits<must_type_kind_t> {
  static void enumeration(IO& io, must_type_kind_t& value) {
    io.enumCase(value, "builtin", BUILTIN);
    io.enumCase(value, "struct", STRUCT);
    io.enumCase(value, "pointer", POINTER);
  }
};

template <>
struct llvm::yaml::MappingTraits<must_type_info_t> {
  static void mapping(IO& io, must_type_info_t& info) {
    io.mapRequired("id", info.id);
    io.mapRequired("kind", info.kind);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(must_type_info_t)

template <>
struct llvm::yaml::MappingTraits<must::StructTypeInfo> {
  static void mapping(IO& io, must::StructTypeInfo& info) {
    io.mapRequired("id", info.id);
    io.mapRequired("name", info.name);
    io.mapRequired("extent", info.extent);
    io.mapRequired("member_count", info.numMembers);
    io.mapRequired("offsets", info.offsets);
    io.mapRequired("types", info.memberTypes);
    io.mapRequired("sizes", info.arraySizes);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(must::StructTypeInfo)

namespace must {

TypeIO::TypeIO(TypeDB& typeDB) : typeDB(typeDB) {
}

bool TypeIO::load(std::string file) {
  using namespace llvm;
  auto memBuffer = MemoryBuffer::getFile(file);

  if (std::error_code ec = memBuffer.getError()) {
    // TODO meaninful error handling/message
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

bool TypeIO::store(std::string file) const {
  using namespace llvm;

  std::error_code ec;
  raw_fd_ostream oss(StringRef(file), ec, sys::fs::OpenFlags::F_Text);

  if (oss.has_error()) {
    llvm::errs() << "Error\n";
    return false;
  }

  yaml::Output out(oss);

  // FIXME why does yaml not cope with const types (only when explicitly registered)
  auto types = typeDB.getStructList();
  out << types;

  return true;
}

}  // namespace must
