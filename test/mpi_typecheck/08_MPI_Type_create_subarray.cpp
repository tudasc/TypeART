// clang-format off
// RUN: %run %s --mpi_intercept --executable %s.exe --command "mpirun -n 2 --output-filename %s.log %s.exe" && cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s && cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include "Util.hpp"

#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  MPI_Datatype mpi_subarray;
  MPI_Type_create_subarray(2, (int[]){3, 3}, (int[]){2, 2}, (int[]){0, 1}, MPI_ORDER_C, MPI_DOUBLE, &mpi_subarray);
  MPI_Type_set_name(mpi_subarray, "test_type");
  MPI_Type_commit(&mpi_subarray);

  double f[9];
  padded_array<8> too_small;

  // clang-format off
  // RANK0: R[0][Info]ID[0] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "test_type"
  // RANK1: R[1][Info]ID[0] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "test_type"
  // clang-format on
  run_test(f, 1, mpi_subarray);

  // clang-format off
  // RANK0: R[0][Info]ID[1] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "test_type"
  // RANK0: R[0][Error]ID[1] buffer too small (8 elements, 9 required)
  // RANK1: R[1][Info]ID[1] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "test_type"
  // RANK1: R[1][Error]ID[1] buffer too small (8 elements, 9 required)
  // clang-format on
  run_test(too_small, 1, mpi_subarray);

  MPI_Type_free(&mpi_subarray);
  MPI_Finalize();
  return 0;
}
