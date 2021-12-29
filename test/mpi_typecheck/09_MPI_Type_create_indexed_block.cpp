
// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan
// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s
// RUN: cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  MPI_Datatype mpi_double_vec;
  MPI_Type_create_indexed_block(3, 2, (int[3]){0, 3, 1}, MPI_DOUBLE, &mpi_double_vec);
  MPI_Type_set_name(mpi_double_vec, "test_type");
  MPI_Type_commit(&mpi_double_vec);

  double f[5];
  padded_array<4> too_small;

  // clang-format off
  // RANK0: R[0][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: successfully checked send-buffer 0x{{.*}} of type [5 x double] against 1 element of MPI type "test_type"
  // RANK1: R[1][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: successfully checked recv-buffer 0x{{.*}} of type [5 x double] against 1 element of MPI type "test_type"
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  run_test(f, 1, mpi_double_vec);

  // clang-format off
  // RANK0: R[0][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: type error while checking send-buffer 0x{{.*}} of type [4 x double] against 1 element of MPI type "test_type": buffer too small (4 elements, 5 required)
  // RANK1: R[1][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: type error while checking recv-buffer 0x{{.*}} of type [4 x double] against 1 element of MPI type "test_type": buffer too small (4 elements, 5 required)
  // clang-format on
  run_test(too_small, 1, mpi_double_vec);

  // RANK0: R[0][Info]T[{{[0-9]*}}] CCounter { Send: 2 Recv: 0 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // RANK1: R[1][Info]T[{{[0-9]*}}] CCounter { Send: 0 Recv: 2 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // CHECK: R[{{0|1}}][Info]T[{{[0-9]*}}] MCounter { Error: 0 Null_Buf: 0 Null_Count: 0 Type_Error: 1 }
  MPI_Type_free(&mpi_double_vec);
  MPI_Finalize();
  return 0;
}
