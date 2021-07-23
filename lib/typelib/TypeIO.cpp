//
// Created by sebastian on 22.03.18.
//

#include "TypeIO.h"

#include "TypeDB.h"
#include "TypeDatabase.h"
#include "support/Logger.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <system_error>
#include <vector>

using namespace llvm::yaml;

template <>
struct ScalarTraits<typeart::StructTypeFlag> {
  static void output(const typeart::StructTypeFlag& value, void*, llvm::raw_ostream& out) {
    out << static_cast<int>(value);
  }

  static StringRef input(StringRef scalar, void*, typeart::StructTypeFlag& value) {
    int flag{0};
    const auto result = scalar.getAsInteger(0, flag);
    if (result) {
      // Error result, assume user_defined:
      value = typeart::StructTypeFlag::USER_DEFINED;
    } else {
      value = static_cast<typeart::StructTypeFlag>(flag);
    }
    return StringRef();
  }

  // Determine if this scalar needs quotes.
  static QuotingType mustQuote(StringRef) {
    return QuotingType::None;
  }
};

template <>
struct llvm::yaml::MappingTraits<typeart::StructTypeInfo> {
  static void mapping(IO& io, typeart::StructTypeInfo& info) {
    io.mapRequired("id", info.id);
    io.mapRequired("name", info.name);
    io.mapRequired("extent", info.extent);
    io.mapRequired("member_count", info.num_members);
    io.mapRequired("offsets", info.offsets);
    io.mapRequired("types", info.member_types);
    io.mapRequired("sizes", info.array_sizes);
    io.mapRequired("flags", info.flag);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(typeart::StructTypeInfo)

namespace typeart {
TypeIO::TypeIO(TypeDB* typeDB) : typeDB(typeDB) {
}

bool TypeIO::load(const std::string& file, std::error_code& ec) {
  using namespace llvm;
  ErrorOr<std::unique_ptr<MemoryBuffer>> memBuffer = MemoryBuffer::getFile(file);

  if (std::error_code error = memBuffer.getError(); error) {
    // TODO meaningful error handling/message
    LOG_WARNING("Warning while loading type file to " << file << ". Reason: " << error.message());
    ec = error;
    return false;
  }

  typeDB->clear();

  yaml::Input in(memBuffer.get()->getMemBufferRef());
  std::vector<StructTypeInfo> structures;
  in >> structures;

  for (auto& typeInfo : structures) {
    typeDB->registerStruct(typeInfo);
  }

  return !in.error();
}

bool TypeIO::store(const std::string& file, std::error_code& ec) const {
  using namespace llvm;

  std::error_code error;
  raw_fd_ostream oss(StringRef(file), ec, sys::fs::OpenFlags::F_Text);

  if (oss.has_error()) {
    LOG_WARNING("Warning while storing type file to " << file << ". Reason: " << ec.message());
    ec = error;
    return false;
  }

  auto types = typeDB->getStructList();
  yaml::Output out(oss);
  if (!types.empty()) {
    out << types;
  } else {
    out.beginDocuments();
    out.endDocuments();
  }

  return true;
}

}  // namespace typeart
