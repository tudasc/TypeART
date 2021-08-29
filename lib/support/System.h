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

#ifndef TYPEART_SYSTEM_H
#define TYPEART_SYSTEM_H

#include <cstdio>
#include <memory>
#include <optional>

namespace typeart {
namespace system {

class Process {
  std::string self_exe;

  Process();

 public:
  static const Process& get();

  [[nodiscard]] const std::string& exe() const;
  [[nodiscard]] static long getMemoryUsage();
};

bool test_command(const std::string& command, const std::string& test_arg = "-h");

}  // namespace system

struct SourceLocation {
  std::string function;
  std::string file;
  std::string line;

  static std::optional<SourceLocation> create(const void* addr);
};

}  // namespace typeart

#endif  // TYPEART_SYSTEM_H
