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

namespace typeart::filter::util {

template <typename T = llvm::json::Value>
inline llvm::Expected<T> getJSON(const llvm::StringRef& src_file) {
  std::string err;
  llvm::raw_string_ostream err_ostream(err);

  if (src_file.empty()) {
    err_ostream << "CG File not set!";
    LOG_FATAL(err_ostream.str());
    std::exit(-1);
  }

  const auto& mem = llvm::MemoryBuffer::getFile(src_file);
  if (!mem) {
    err_ostream << mem.getError().message();
    LOG_FATAL(err_ostream.str());
    std::exit(-1);
  }

  // it's easier to not use llvm::json::parse<T>() here
  auto parsed_json = llvm::json::parse(mem.get()->getBuffer());
  if (!parsed_json) {
    llvm::logAllUnhandledErrors(parsed_json.takeError(), err_ostream);
    LOG_FATAL(err_ostream.str());
    std::exit(-1);
  }

  if constexpr (std::is_same_v<llvm::json::Value, T>) {
    return parsed_json;
  } else {
#if LLVM_VERSION_MAJOR < 12
    T result;
    if (fromJSON(*parsed_json, result)) {
      return std::move(result);
    }
    LOG_FATAL("invalid json");
    std::exit(-1);
#else
    llvm::json::Path::Root root("");
    T result;
    if (fromJSON(*parsed_json, result, root)) {
      return std::move(result);
    }
    llvm::logAllUnhandledErrors(root.getError(), err_ostream);
    LOG_FATAL(err_ostream.str());
    std::exit(-1);
#endif
  }
}

}  // namespace typeart::filter::util

#endif  // TYPEART_FILTER_JSON_H
