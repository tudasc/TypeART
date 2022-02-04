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

#ifndef TYPEART_SYSTEM_H
#define TYPEART_SYSTEM_H

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace typeart {
namespace system {

class Process {
  std::string self_exe;

  Process();

 public:
  static const Process& get();

  [[nodiscard]] const std::string& exe() const;
  [[nodiscard]] static long getMaxRSS();
};

bool test_command(std::string_view command, std::string_view test_arg = "-h");

}  // namespace system

struct BinaryLocation {
  std::string file;
  void* file_addr;
  std::optional<std::string> function;
  void* function_addr;

  static std::optional<BinaryLocation> create(const void* addr);
};

struct SourceLocation {
  std::string function;
  std::string file;
  std::string line;

  static std::optional<SourceLocation> create(const void* addr);
};

constexpr size_t MAX_STACKTRACE_SIZE = 512;

struct StacktraceEntry {
  void* addr;
  std::optional<BinaryLocation> binary;
  std::optional<SourceLocation> source;

  static StacktraceEntry create(void* addr);
};

std::ostream& operator<<(std::ostream& os, const StacktraceEntry& entry);

class Stacktrace {
  using value_type = StacktraceEntry;

  std::vector<StacktraceEntry> entries;

  Stacktrace(std::vector<StacktraceEntry> entries);

 public:
  static Stacktrace current();

  inline auto begin() {
    return entries.begin();
  }

  inline auto end() {
    return entries.end();
  }

  inline auto begin() const {
    return entries.begin();
  }

  inline auto end() const {
    return entries.end();
  }
};

}  // namespace typeart

#endif  // TYPEART_SYSTEM_H
