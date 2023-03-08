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

#ifndef TYPEART_JSON_H
#define TYPEART_JSON_H

#include "Error.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>

namespace typeart::util {

template <typename T = llvm::json::Value>
inline llvm::Expected<T> getJSON(const llvm::StringRef& SrcFile) {
  std::string Err;
  llvm::raw_string_ostream ErrOStream(Err);

  if (SrcFile.empty()) {
    ErrOStream << "CG File not set!";
    LOG_FATAL(ErrOStream.str());
    std::exit(-1);
  }

  const auto& Mem = llvm::MemoryBuffer::getFile(SrcFile);
  if (!Mem) {
    ErrOStream << Mem.getError().message();
    LOG_FATAL(ErrOStream.str());
    std::exit(-1);
  }

  if constexpr (std::is_same_v<llvm::json::Value, T>) {
    auto ParsedJSON = llvm::json::parse(Mem.get()->getBuffer());
    if (!ParsedJSON) {
      llvm::logAllUnhandledErrors(ParsedJSON.takeError(), ErrOStream);
      LOG_FATAL(ErrOStream.str());
      std::exit(-1);
    }

    return ParsedJSON;
  } else {
    auto ParsedJSON = llvm::json::parse<T>(Mem.get()->getBuffer());
    if (!ParsedJSON) {
      llvm::logAllUnhandledErrors(ParsedJSON.takeError(), ErrOStream);
      LOG_FATAL(ErrOStream.str());
      std::exit(-1);
    }

    return ParsedJSON;
  }
}

}  // namespace typeart::util

#endif  // TYPEART_JSON_H
