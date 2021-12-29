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
  // RANK0: R[0][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: successfully checked send-buffer 0x{{.*}} of type [1 x struct.A] against 16 elements of MPI type "MPI_DOUBLE"
  // RANK1: R[1][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: successfully checked recv-buffer 0x{{.*}} of type [1 x struct.A] against 16 elements of MPI type "MPI_DOUBLE"
  // clang-format on
  run_test(a.arr, n, MPI_DOUBLE);

  // clang-format off
  // RANK0: R[0][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: successfully checked send-buffer 0x{{.*}} of type [1 x struct.B] against 16 elements of MPI type "MPI_DOUBLE"
  // RANK1: R[1][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: successfully checked recv-buffer 0x{{.*}} of type [1 x struct.B] against 16 elements of MPI type "MPI_DOUBLE"
  // clang-format on
  run_test(b.a.arr, n, MPI_DOUBLE);

  // RANK0: R[0][Info]T[{{[0-9]*}}] CCounter { Send: 2 Recv: 0 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // RANK1: R[1][Info]T[{{[0-9]*}}] CCounter { Send: 0 Recv: 2 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // CHECK: R[{{0|1}}][Info]T[{{[0-9]*}}] MCounter { Error: 0 Null_Buf: 0 Null_Count: 0 Type_Error: 0 }
  MPI_Finalize();
  return 0;
}
