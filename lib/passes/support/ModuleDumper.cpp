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

#include "ModuleDumper.h"

#include "support/Logger.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <string_view>

namespace typeart::util::module::detail {

void dump_module(const llvm::Module& module, llvm::raw_ostream& out_s) {
  module.print(out_s, nullptr);
}

std::string_view gets_source_file(const llvm::Module& module) {
  return module.getSourceFileName();
}

std::string_view get_file_ext(ModulePhase phase) {
  switch (phase) {
    case ModulePhase::kBase:
      return "_base.ll";
    case ModulePhase::kHeap:
      return "_heap.ll";
    case ModulePhase::kOpt:
      return "_opt.ll";
    case ModulePhase::kStack:
      return "_stack.ll";
  }
  return "_unknown.ll";
}

}  // namespace typeart::util::module::detail

namespace typeart::util::module {

void dump_module(const llvm::Module& module, ModulePhase phase) {
  if (std::getenv("TYPEART_PASS_INTERNAL_EMIT_IR") == nullptr) {
    LOG_DEBUG("No dump required")
    return;
  }

  const auto source = std::filesystem::path{detail::gets_source_file(module)};

  if (!source.has_filename()) {
    LOG_ERROR("No filename for module")
    return;
  }

  const auto source_ll = source.parent_path() / (source.stem().string() + detail::get_file_ext(phase).data());
  LOG_DEBUG("Dumping to file " << source_ll);

  std::error_code error_code;
  llvm::raw_fd_ostream file_out{source_ll.c_str(), error_code};
  if (error_code) {
    LOG_FATAL("Error while opening file " << error_code.message())
    return;
  }

  detail::dump_module(module, file_out);
  file_out.close();
}

}  // namespace typeart::util::module