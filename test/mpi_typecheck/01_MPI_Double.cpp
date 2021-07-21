// clang-format off
// RUN: %run %s --mpi_intercept --mpi_output_filename "%s.log" && cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s && cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include <mpi.h>
#include <stdlib.h>

const auto n = 16;

struct padded_array {
  double offset;  // needed, cause otherwise the TypeART type of arr will be struct
  double arr[n];
  double padding;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  padded_array f;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    // RANK0: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Send(f.arr, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    // RANK1: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Recv(f.arr, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank == 0) {
    // clang-format off
    // RANK0: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    // RANK0: R[0][Error][1] MPI_Send: buffer 0x{{.*}} too small. The buffer can only hold 16 elements (17 required)
    // clang-format on
    MPI_Send(f.arr, n + 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    // clang-format off
    // RANK1: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    // RANK1: R[1][Error][0] MPI_Recv: buffer 0x{{.*}} too small. The buffer can only hold 16 elements (17 required)
    // clang-format on
    MPI_Recv(f.arr, n + 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Finalize();
  return 0;
}
