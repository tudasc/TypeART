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
  const auto rStr = std::getenv("TYPEART_MPI_LOG");
  std::vector<int> ranks{};

  if (rStr == nullptr) {
    ranks.push_back(0);
    return ranks;
  }

  if (strncmp(rStr, "all", 3) == 0) {
    return ranks;
  }

  ranks.push_back(std::atoi(rStr));
  return ranks;
}

void mpi_log(const std::string& msg) {
  int initFlag{0};
  int finiFlag{0};
  MPI_Initialized(&initFlag);
  MPI_Finalized(&finiFlag);

  if (initFlag != 0 && finiFlag == 0) {
    int mRank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
    const auto outputRanks = getRanks();

    const auto isIn = [&](int r) {
      if (outputRanks.empty()) {
        return true;
      }
      return std::find(std::begin(outputRanks), std::end(outputRanks), r) != std::end(outputRanks);
    };

    if (isIn(mRank)) {
      llvm::errs() << "R[" << mRank << "]" << msg;
    }
  } else {
    llvm::errs() << msg;
  }
}

}  // namespace typeart::detail