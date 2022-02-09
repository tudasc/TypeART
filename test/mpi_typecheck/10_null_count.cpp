// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan
// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | %filecheck --check-prefixes CHECK,RANK0 %s
// RUN: cat "%s.log/1/rank.1/stderr" | %filecheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  double f[4];

  // clang-format off
  // RANK0: R[0][Warning]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: attempted to send 0 elements of buffer 0x{{.*}} 
  // RANK1: R[1][Warning]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: attempted to receive 0 elements of buffer 0x{{.*}} 
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  run_test(f, 0, MPI_DOUBLE);

  // RANK0: R[0][Info]T[{{[0-9]*}}] CCounter { Send: 1 Recv: 0 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // RANK1: R[1][Info]T[{{[0-9]*}}] CCounter { Send: 0 Recv: 1 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // CHECK: R[{{0|1}}][Info]T[{{[0-9]*}}] MCounter { Error: 0 Null_Buf: 0 Null_Count: 1 Type_Error: 0 }
  MPI_Finalize();
  return 0;
}
