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
  // RANK0:  MPI_Send: internal error{{.*}} combiner MPI_Type_create_hvector is currently not supported
  // RANK1: [Test] Received: 1, 4, 7.
  if (rank == 0) {
    MPI_Datatype hvec_type;
    MPI_Type_create_hvector(3, 1, 3 * sizeof(int), MPI_INT, &hvec_type);
    MPI_Type_commit(&hvec_type);

    int buffer[3][3] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    MPI_Send(&buffer[0][1], 1, hvec_type, 1, 0, MPI_COMM_WORLD);

    MPI_Type_free(&hvec_type);
  } else {
    int received[3];
    MPI_Recv(&received, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    fprintf(stderr, "[Test] Received: %d, %d, %d.\n", received[0], received[1], received[2]);
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
