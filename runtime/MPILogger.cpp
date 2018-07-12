#include "Logger.h"
#include "mpi.h"

#include <cstdlib>
#include <vector>

auto getRanks() {
  auto rStr = std::getenv("TYPEART_MPI_LOG");

	std::vector<int> ranks;
  if (rStr == nullptr) {
    ranks.push_back(0);
		return ranks;
	}

	ranks.push_back(std::atoi(rStr));
	return ranks;
}



void mpi_log(int lvl, std::string lvlName, std::string fileName, std::string func, long int line, std::string msg) {
	int initFlag = 0;
	int finiFlag = 0;
	MPI_Initialized(&initFlag);
	MPI_Finalized(&finiFlag);

  if (initFlag && !finiFlag) {
	  int mRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
    auto outputRanks = getRanks();

    const auto isIn = [&](int r) {
      return std::find(std::begin(outputRanks), std::end(outputRanks), r) != std::end(outputRanks);
    };

    if (isIn(mRank)) {
      llvm::errs() << "Message from rank " << mRank << ":\n";
      llvm::errs() << lvlName << " " << fileName << ":" << func << ":" << line << ":" << msg << "\n";
    }
	} else {
		llvm::errs() << lvlName << " " << fileName << ":" << func << ":" << line << ":" << msg << "\n";
	}
}
