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

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <mpi.h>
#include <string>
#include <vector>

namespace typeart::detail {

inline std::vector<int> getRanks() {
  const auto* rank_option = std::getenv("TYPEART_MPI_LOG");
  std::vector<int> ranks{};

  if (rank_option == nullptr) {
    ranks.push_back(0);
    return ranks;
  }

  if (strncmp(rank_option, "all", 3) == 0) {
    return ranks;
  }

  ranks.push_back(std::atoi(rank_option));
  return ranks;
}

void typeart_log(const std::string& msg) {
  int mpi_initialized{0};
  int mpi_finalized{0};
  MPI_Initialized(&mpi_initialized);
  MPI_Finalized(&mpi_finalized);

  if (mpi_initialized != 0 && mpi_finalized == 0) {
    int mpi_rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    const auto outputRanks = getRanks();

    const auto isIn = [&](int rank) {
      if (outputRanks.empty()) {
        return true;
      }
      return std::find(std::begin(outputRanks), std::end(outputRanks), rank) != std::end(outputRanks);
    };

    if (isIn(mpi_rank)) {
      llvm::errs() << "R[" << mpi_rank << "]" << msg;
    }
  } else {
    llvm::errs() << msg;
  }
}

}  // namespace typeart::detail