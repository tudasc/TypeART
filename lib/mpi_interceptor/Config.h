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

#ifndef TYPEART_MPI_INTERCEPTOR_CONFIG_H
#define TYPEART_MPI_INTERCEPTOR_CONFIG_H

namespace typeart {

class Config {
 public:
  enum class SourceLocation { None, Error, All };

 private:
  bool with_backtraces;
  SourceLocation source_location;

  Config();

 public:
  static const Config& get() {
    static Config instance;
    return instance;
  }

  bool isWithBacktraces() const {
    return with_backtraces;
  }

  SourceLocation getSourceLocation() const {
    return source_location;
  }
};

}  // namespace typeart

#endif  // TYPEART_MPI_INTERCEPTOR_CONFIG_H
