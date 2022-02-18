// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Config.h"

#include <cstdlib>
#include <cstring>
#include <utility>

namespace typeart {

template <class... String>
bool strcmp_any_of(const char* lhs, String... rhs) {
  return lhs != nullptr && ((std::strcmp(lhs, rhs) == 0) || ...);
}

Config::Config() {
  with_backtraces = strcmp_any_of(std::getenv("TYPEART_STACKTRACE"), "1", "ON");

  auto source_location_env = std::getenv("TYPEART_SOURCE_LOCATION");

  if (strcmp_any_of(source_location_env, "error", "some")) {
    source_location = SourceLocation::Error;
  } else if (strcmp_any_of(source_location_env, "ON", "all")) {
    source_location = SourceLocation::All;
  } else {
    source_location = SourceLocation::None;
  }
}

}  // namespace typeart
