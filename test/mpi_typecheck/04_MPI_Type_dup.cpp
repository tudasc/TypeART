// clang-format off
// RUN: %run %s --mpi_intercept --mpi_output_filename "%s.log" && cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s && cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include "Util.hpp"

#include <mpi.h>
#include <stdlib.h>

constexpr auto n = 16;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  MPI_Datatype mpi_double;
  MPI_Type_dup(MPI_DOUBLE, &mpi_double);
  MPI_Type_set_name(mpi_double, "test_type");
  MPI_Type_commit(&mpi_double);

  // CHECK: [Trace] TypeART Runtime Trace

  double f[n];
  padded_array<n> too_small;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // clang-format off
  // RANK0: R[0][Info]ID[0] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "test_type"
  // RANK1: R[1][Info]ID[0] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "test_type"
  // clang-format on
  run_test(f, n, mpi_double);

  // clang-format off
  // RANK0: R[0][Info]ID[1] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "test_type"
  // RANK0: R[0][Error]ID[1] buffer too small (16 elements, 17 required)
  // RANK1: R[1][Info]ID[1] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "test_type"
  // RANK1: R[1][Error]ID[1] buffer too small (16 elements, 17 required)
  // clang-format on
  run_test(too_small, n + 1, mpi_double);

  MPI_Type_free(&mpi_double);
  MPI_Finalize();
  return 0;
}
