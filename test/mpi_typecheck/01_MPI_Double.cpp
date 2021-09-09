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

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  padded_array<n> data;

  // clang-format off
  // RANK0: R[0][Info]ID[0] run_test(void*, int, {{.*}}[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Send: checking send-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // RANK1: R[1][Info]ID[0] run_test(void*, int, {{.*}}[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Recv: checking recv-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  run_test(data, n, MPI_DOUBLE);

  // clang-format off
  // RANK0: R[0][Info]ID[1] run_test(void*, int, {{.*}}[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Send: checking send-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // RANK0: R[0][Error]ID[1] buffer too small (16 elements, 17 required)
  // RANK1: R[1][Info]ID[1] run_test(void*, int, {{.*}}[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Recv: checking recv-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // RANK1: R[1][Error]ID[1] buffer too small (16 elements, 17 required)
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  run_test(data, n + 1, MPI_DOUBLE);

  // clang-format off
  // RANK0: R[0][Info]ID[2] main[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Sendrecv: checking send-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // RANK0: R[0][Info]ID[3] main[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Sendrecv: checking recv-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // RANK1: R[1][Info]ID[2] main[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Sendrecv: checking send-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // RANK1: R[1][Info]ID[3] main[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Sendrecv: checking recv-buffer 0x{{.*}} of type "double" against MPI type "MPI_DOUBLE"
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    MPI_Sendrecv(data.arr, n, MPI_DOUBLE, 1, 0, data.arr, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    MPI_Sendrecv(data.arr, n, MPI_DOUBLE, 0, 0, data.arr, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // RANK0: R[0][Info] CCounter { Send: 3 Recv: 1 Send_Recv: 1 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // RANK1: R[1][Info] CCounter { Send: 1 Recv: 3 Send_Recv: 1 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // CHECK: R[{{0|1}}][Info] MCounter { Error: 0 Null_Buf: 0 Null_Count: 0 Type_Error: 1 }
  MPI_Finalize();
  return 0;
}
