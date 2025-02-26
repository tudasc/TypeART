// TypeART library
//
// Copyright (c) 2017-2025 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "support/Logger.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <optional>

namespace typeart::error {

#define RETURN_NONE_IF(condition, ...)                \
  if (condition) {                                    \
    std::string message = llvm::formatv(__VA_ARGS__); \
    LOG_DEBUG(message);                               \
    return {};                                        \
  }

#define RETURN_ON_NONE(expr) \
  if (!(expr)) {             \
    return {};               \
  }

#define RETURN_ON_ERROR(expr)          \
  if (auto err = (expr).takeError()) { \
    return {std::move(err)};           \
  }

inline llvm::Error make_string_error(const char* message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
  // return llvm::make_error<llvm::StringError>(llvm::inconvertibleErrorCode(), message);
}

inline llvm::Error make_string_error(const std::string& message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
  // return llvm::make_error<llvm::StringError>( message);
}

#define RETURN_ERROR_IF(condition, ...)                          \
  if (condition) {                                               \
    std::string message = llvm::formatv(__VA_ARGS__);            \
    LOG_FATAL(message);                                          \
    return {typeart::error::make_string_error(message.c_str())}; \
  }

}  // namespace typeart::error
