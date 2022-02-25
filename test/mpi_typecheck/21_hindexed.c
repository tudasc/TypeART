// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | %filecheck --check-prefixes CHECK,RANK0 %s
// RUN: cat "%s.log/1/rank.1/stderr" | %filecheck --check-prefixes CHECK,RANK1 %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void run(int rank) {
  // RANK0:  MPI_Send: internal error{{.*}} combiner MPI_Type_create_hindexed is currently not supported
  // RANK1: [Test] Received: 0 3 4 7 8
  if (rank == 0) {
    MPI_Datatype index_type;
    int lengths[3]            = {1, 2, 2};
    MPI_Aint displacements[3] = {0, 3 * sizeof(int), 7 * sizeof(int)};
    MPI_Type_create_hindexed(3, lengths, displacements, MPI_INT, &index_type);
    MPI_Type_commit(&index_type);

    int buffer[3][3] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    MPI_Send(buffer, 1, index_type, 1, 0, MPI_COMM_WORLD);

    MPI_Type_free(&index_type);
  } else {
    const int length = 5;
    int received[length];
    MPI_Recv(&received, length, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    fprintf(stderr, "[Test] Received:");
    for (int i = 0; i < length; ++i) {
      fprintf(stderr, " %i", received[i]);
    }
    fprintf(stderr, "\n");
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  run(rank);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
