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

  MPI_Datatype mpi_double;
  MPI_Type_dup(MPI_DOUBLE, &mpi_double);
  MPI_Type_commit(&mpi_double);

  // CHECK: [Trace] TypeART Runtime Trace

  padded_array f2;
  double f[n];
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    // RANK0: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Send(f, n, mpi_double, 1, 0, MPI_COMM_WORLD);
  } else {
    // RANK1: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Recv(f, n, mpi_double, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank == 0) {
    // clang-format off
    // RANK0: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    // RANK0: R[0][Error][1] MPI_Send: buffer 0x{{.*}} too small. The buffer can only hold 16 elements (17 required)
    // clang-format on
    MPI_Send(f2.arr, n + 1, mpi_double, 1, 0, MPI_COMM_WORLD);
  } else {
    // clang-format off
    // RANK1: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    // RANK1: R[1][Error][0] MPI_Recv: buffer 0x{{.*}} too small. The buffer can only hold 16 elements (17 required)
    // clang-format on
    MPI_Recv(f2.arr, n + 1, mpi_double, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Type_free(&mpi_double);
  MPI_Finalize();
  return 0;
}
