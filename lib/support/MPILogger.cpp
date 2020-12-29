#include "Logger.h"
#include "mpi.h"

#include <cstdlib>
#include <vector>

inline std::vector<int> getRanks() {
  const auto rStr = std::getenv("TYPEART_MPI_LOG");

  std::vector<int> ranks;
  if (rStr == nullptr) {
    ranks.push_back(0);
    return ranks;
  } else if (strncmp(rStr, "all", 3) == 0) {
    return ranks;
  }

  ranks.push_back(std::atoi(rStr));
  return ranks;
}

void mpi_log(const std::string& msg) {
  int initFlag = 0;
  int finiFlag = 0;
  MPI_Initialized(&initFlag);
  MPI_Finalized(&finiFlag);

  if (initFlag != 0 && finiFlag == 0) {
    int mRank;
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
