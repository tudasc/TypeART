//
// Created by sebastian on 13.01.21.
//

#include "Runtime.h"

#include "AccessCountPrinter.h"
#include "AccessCounter.h"
#include "RuntimeData.h"
#include "TypeIO.h"
#include "support/Logger.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

namespace typeart {

namespace debug {

std::string toString(const void* memAddr, int typeId, size_t count, size_t typeSize, const void* calledFrom) {
  std::string buf;
  llvm::raw_string_ostream s(buf);
  const auto name = typeart::RuntimeSystem::get().typeResolution.getTypeName(typeId);
  s << memAddr << " " << typeId << " " << name << " " << typeSize << " " << count << " (" << calledFrom << ")";
  return s.str();
}

std::string toString(const void* memAddr, int typeId, size_t count, const void* calledFrom) {
  const auto typeSize = typeart::RuntimeSystem::get().typeResolution.getTypeSize(typeId);
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

  auto loadTypes = [this](const std::string& file) -> bool {
    TypeIO cio(typeDB);
    return cio.load(file);
  };

  // Try to load types from specified file first.
  // Then look at default location.
  const char* typeFile = std::getenv("TA_TYPE_FILE");
  if (typeFile != nullptr) {
    if (!loadTypes(typeFile)) {
      LOG_FATAL("Failed to load recorded types from " << typeFile);
      std::exit(EXIT_FAILURE);  // TODO: Error handling
    }
  } else {
    if (!loadTypes(defaultTypeFileName)) {
      LOG_FATAL("No type file with default name \""
                << defaultTypeFileName
                << "\" in current directory. To specify a different file, edit the TA_TYPE_FILE environment variable.");
      std::exit(EXIT_FAILURE);  // TODO: Error handling
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

  std::string stats;
  llvm::raw_string_ostream stream(stats);
  softcounter::serialise(recorder, stream);
  if (!stream.str().empty()) {
    // llvm::errs/LOG will crash with virtual call error
    std::cerr << stream.str();
  }
}

// This is initially set to true in order to prevent tracking anything before the runtime library is properly set up.
thread_local bool RuntimeSystem::rtScope = false;

}  // namespace typeart
