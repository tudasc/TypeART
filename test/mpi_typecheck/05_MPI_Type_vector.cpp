// clang-format off
// RUN: %run %s --mpi_intercept --mpi_output_filename "%s.log" && cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s && cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include "Util.hpp"

#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  MPI_Datatype mpi_double_vec;
  MPI_Type_vector(3, 2, 3, MPI_DOUBLE, &mpi_double_vec);
  MPI_Type_set_name(mpi_double_vec, "test_type");
  MPI_Type_commit(&mpi_double_vec);

  double f[8];
  padded_array<7> too_small;

  // clang-format off
  // RANK0: R[0][Info]ID[0] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "test_type"
  // RANK1: R[1][Info]ID[0] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "test_type"
  // clang-format on
  run_test(f, 1, mpi_double_vec);

  // clang-format off
  // RANK0: R[0][Info]ID[1] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "test_type"
  // RANK0: R[0][Error]ID[1] buffer too small (7 elements, 8 required)
  // RANK1: R[1][Info]ID[1] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "test_type"
  // RANK1: R[1][Error]ID[1] buffer too small (7 elements, 8 required)
  // clang-format on
  run_test(too_small, 1, mpi_double_vec);

  MPI_Type_free(&mpi_double_vec);
  MPI_Finalize();
  return 0;
}
