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

#ifndef TYPEART_FILECONFIGURATION_H
#define TYPEART_FILECONFIGURATION_H

#include "support/Configuration.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorOr.h"

namespace typeart::config {
struct TypeARTConfigOptions;
}

namespace typeart::config::file {

// using FileOptionsMap = llvm::StringMap<config::OptionValue>;

class FileOptions : public config::Configuration {
 public:
  [[nodiscard]] std::optional<config::OptionValue> getValue(std::string_view opt_path) const override = 0;
  [[nodiscard]] virtual OptionsMap getConfiguration() const                                           = 0;
  [[nodiscard]] virtual std::string getConfigurationAsString() const                                  = 0;
  ~FileOptions() override                                                                             = default;
};

[[maybe_unused]] llvm::ErrorOr<std::unique_ptr<FileOptions>> make_file_configuration(std::string_view file_path);

[[maybe_unused]] llvm::ErrorOr<std::unique_ptr<FileOptions>> make_default_file_configuration();
[[maybe_unused]] llvm::ErrorOr<std::unique_ptr<FileOptions>> make_from_configuration(
    const TypeARTConfigOptions& options);

[[maybe_unused]] llvm::ErrorOr<bool> write_file_configuration(llvm::raw_ostream& out_stream,
                                                              const FileOptions& file_options);

}  // namespace typeart::config::file

#endif  // TYPEART_FILECONFIGURATION_H
