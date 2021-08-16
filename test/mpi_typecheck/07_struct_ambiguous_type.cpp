// clang-format off
// RUN: %run %s --mpi_intercept --mpi_output_filename "%s.log" && cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s && cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include "Util.hpp"

#include <mpi.h>
#include <stdlib.h>

constexpr auto n = 16;

struct A {
  double arr[16];
};

struct B {
  A a;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  A a;
  B b;

  // clang-format off
  // RANK0: R[0][Info]ID[0] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "MPI_DOUBLE"
  // RANK0: R[0][Error]ID[0] expected a type matching MPI type "MPI_DOUBLE", but found type "struct.A"
  // RANK0: R[0][Info]ID[0] found struct member at offset 0 with type "double", checking with this type...
  // RANK1: R[1][Info]ID[0] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "MPI_DOUBLE"
  // RANK1: R[1][Error]ID[0] expected a type matching MPI type "MPI_DOUBLE", but found type "struct.A"
  // RANK1: R[1][Info]ID[0] found struct member at offset 0 with type "double", checking with this type...
  // clang-format on
  run_test(a.arr, n, MPI_DOUBLE);

  // clang-format off
  // RANK0: R[0][Info]ID[1] MPI_Send at 0x{{.*}} in function _Z8run_test{{.*}}: checking send-buffer 0x{{.*}} against MPI type "MPI_DOUBLE"
  // RANK0: R[0][Error]ID[1] expected a type matching MPI type "MPI_DOUBLE", but found type "struct.B"
  // RANK0: R[0][Info]ID[1] found struct member at offset 0 with type "struct.A", checking with this type...
  // RANK0: R[0][Error]ID[1] expected a type matching MPI type "MPI_DOUBLE", but found type "struct.A"
  // RANK0: R[0][Info]ID[1] found struct member at offset 0 with type "double", checking with this type...
  // RANK1: R[1][Info]ID[1] MPI_Recv at 0x{{.*}} in function _Z8run_test{{.*}}: checking recv-buffer 0x{{.*}} against MPI type "MPI_DOUBLE"
  // RANK1: R[1][Error]ID[1] expected a type matching MPI type "MPI_DOUBLE", but found type "struct.B"
  // RANK1: R[1][Info]ID[1] found struct member at offset 0 with type "struct.A", checking with this type...
  // RANK1: R[1][Error]ID[1] expected a type matching MPI type "MPI_DOUBLE", but found type "struct.A"
  // RANK1: R[1][Info]ID[1] found struct member at offset 0 with type "double", checking with this type...
  // clang-format on
  run_test(b.a.arr, n, MPI_DOUBLE);

  MPI_Finalize();
  return 0;
}