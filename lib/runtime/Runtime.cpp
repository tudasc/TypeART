// TypeART library
//
// Copyright (c) 2017-2022 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Runtime.h"

#include "AccessCountPrinter.h"
#include "AccessCounter.h"
#include "RuntimeData.h"
#include "TypeIO.h"
#include "support/Logger.h"

//#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace typeart {

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom) {
  std::string buf;
  llvm::raw_string_ostream s(buf);
  const auto name = typeart::RuntimeSystem::get().typeResolution.db().getTypeName(typeId);
  s << memAddr << " " << typeId << " " << name << " " << typeSize << " " << count << " (" << calledFrom << ")";
  return s.str();
}

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom) {
  const auto typeSize = typeart::RuntimeSystem::get().typeResolution.db().getTypeSize(typeId);
  return toString(memAddr, typeId, count, typeSize, calledFrom);
}

std::string toString(const void* addr, const PointerInfo& info) {
  return toString(addr, info.typeId, info.count, info.debug);
}

inline void printTraceStart() {
  LOG_TRACE("TypeART Runtime Trace");
  LOG_TRACE("*********************");
  LOG_TRACE("Operation  Address   Type   Size   Count   (CallAddr)   Stack/Heap/Global");
  LOG_TRACE("-------------------------------------------------------------------------");
}

}  // namespace debug

static constexpr const char* defaultTypeFileName = "types.yaml";

RuntimeSystem::RuntimeSystem() : rtScopeInit(), typeResolution(typeDB, recorder), allocTracker(typeDB, recorder) {
  debug::printTraceStart();

  auto loadTypes = [this](const std::string& file, std::error_code& ec) -> bool {
    auto loaded = io::load(&typeDB, file);
    ec          = loaded.getError();
    return !static_cast<bool>(ec);
  };

  std::error_code error;
  // Try to load types from specified file first.
  // Then look at default location.
  const char* type_file = std::getenv("TYPEART_TYPE_FILE");
  if (type_file == nullptr) {
    // FIXME Deprecated name
    type_file = std::getenv("TA_TYPE_FILE");
    if (type_file != nullptr) {
      LOG_WARNING("Use of deprecated env var TA_TYPE_FILE.");
    }
  }
  if (type_file != nullptr) {
    if (!loadTypes(type_file, error)) {
      LOG_FATAL("Failed to load recorded types from TYPEART_TYPE_FILE=" << type_file
                                                                        << ". Reason: " << error.message());
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  } else {
    if (!loadTypes(defaultTypeFileName, error)) {
      LOG_WARNING(
          "No type file with default name \""
          << defaultTypeFileName
          << "\" in current directory. Using default built-in types only. To specify a different file, edit the "
             "TYPEART_TYPE_FILE environment variable. Reason: "
          << error.message());
    }
  }

  std::stringstream ss;
  const auto& typeList = typeDB.getStructList();
  for (const auto& structInfo : typeList) {
    ss << structInfo.name << ", ";
  }
  recorder.incUDefTypes(typeList.size());
  LOG_INFO("Recorded types: " << ss.str());
  rtScopeInit.reset();
}

RuntimeSystem::~RuntimeSystem() {
  rtScope = true;

  //  std::string stats;
  //  llvm::raw_string_ostream stream(stats);

  std::ostringstream stream;
  softcounter::serialize(recorder, stream);
  if (!stream.str().empty()) {
    // llvm::errs/LOG will crash with virtual call error
    std::cerr << stream.str();
  }
}

// This is initially set to true in order to prevent tracking anything before the runtime library is properly set up.
thread_local bool RuntimeSystem::rtScope = false;

}  // namespace typeart
