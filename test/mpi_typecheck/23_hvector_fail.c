// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 %s.exe" 2>&1 | %filecheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void run(int rank) {
  // clang-format off
  // CHECK: MPI_Send: type error{{.*}} type [8 x int32] against 1 element of MPI type "MPI_Type_create_hvector": buffer too small (8 elements, 13 required)
  // clang-format on
  if (rank == 0) {
    MPI_Datatype hvec_type;
    MPI_Type_create_hvector(3, 1, 3 * sizeof(double), MPI_INT, &hvec_type);
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
