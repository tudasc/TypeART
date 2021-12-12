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

#ifndef TYPEART_MPI_INTERCEPTOR_STATS_H
#define TYPEART_MPI_INTERCEPTOR_STATS_H

#include <atomic>

namespace typeart {

struct CallCounter {
  std::atomic_size_t send        = {0};
  std::atomic_size_t recv        = {0};
  std::atomic_size_t send_recv   = {0};
  std::atomic_size_t unsupported = {0};
};

struct MPICounter {
  std::atomic_size_t null_count = {0};
  std::atomic_size_t null_buff  = {0};
  std::atomic_size_t type_error = {0};
  std::atomic_size_t error      = {0};
};

}  // namespace typeart

#endif  // TYPEART_MPI_INTERCEPTOR_STATS_H
