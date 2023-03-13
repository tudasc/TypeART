// TypeART library
//
// Copyright (c) 2017-2023 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#ifndef TYPEART_FILTER_JSON_H
#define TYPEART_FILTER_JSON_H

#include "support/Error.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>

namespace typeart::filter {

template <typename T = llvm::json::Value>
inline llvm::Expected<T> getJSON(const llvm::StringRef& srcFile) {
  std::string err;
  llvm::raw_string_ostream errOStream(err);

  if (srcFile.empty()) {
    errOStream << "CG File not set!";
    LOG_FATAL(errOStream.str());
    std::exit(-1);
  }

  const auto& mem = llvm::MemoryBuffer::getFile(srcFile);
  if (!mem) {
    errOStream << mem.getError().message();
    LOG_FATAL(errOStream.str());
    std::exit(-1);
  }

  // it's easier to not use llvm::json::parse<T>() here
  auto parsedJSON = llvm::json::parse(mem.get()->getBuffer());
  if (!parsedJSON) {
    llvm::logAllUnhandledErrors(parsedJSON.takeError(), errOStream);
    LOG_FATAL(errOStream.str());
    std::exit(-1);
  }

  if constexpr (std::is_same_v<llvm::json::Value, T>) {
    return parsedJSON;
  } else {
#if LLVM_VERSION_MAJOR < 12
    T result;
    if (fromJSON(*parsedJSON, result)) {
      return std::move(result);
    }
    LOG_FATAL("invalid json");
    std::exit(-1);
#else
    llvm::json::Path::Root root("");
    T result;
    if (fromJSON(*parsedJSON, result, root)) {
      return std::move(result);
    }
    llvm::logAllUnhandledErrors(root.getError(), errOStream);
    LOG_FATAL(errOStream.str());
    std::exit(-1);
#endif
  }
}

}  // namespace typeart::filter

#endif  // TYPEART_FILTER_JSON_H
