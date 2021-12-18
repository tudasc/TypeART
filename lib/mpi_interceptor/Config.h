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
  Config();

 public:
  bool with_backtraces;

  enum class SourceLocation { None, Error, All };
  SourceLocation source_location;


 public:
  static const Config& get() {
    static Config instance;
    return instance;
  }
};

}  // namespace typeart

#endif  // TYPEART_MPI_INTERCEPTOR_CONFIG_H
