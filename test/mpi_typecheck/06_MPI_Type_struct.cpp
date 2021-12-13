// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan
// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s
// RUN: cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include <mpi.h>

struct S1 {
  double a[2];
  int b;
  double c;
};

void run_test(void* data, int count, int integers[], MPI_Aint addrs[], MPI_Datatype types[]) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Datatype datatype;
  MPI_Type_create_struct(count, integers, addrs, types, &datatype);
  MPI_Type_set_name(datatype, "test_type");
  MPI_Type_commit(&datatype);
  if (rank == 0) {
    MPI_Send(data, 1, datatype, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(data, 1, datatype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Type_free(&datatype);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int counts[3]         = {2, 1, 1};
  MPI_Aint offsets[3]   = {offsetof(S1, a), offsetof(S1, b), offsetof(S1, c)};
  MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_DOUBLE};

  double arr[3];
  S1 s1;

  // 1: Check non-struct buffer type and wrong member count
  // clang-format off
  // RANK0: R[0][Info]ID[0] MPI_Send at 0x{{.*}}: checking send-buffer 0x{{.*}} of type "double" against MPI type "test_type"
  // RANK0: R[0][Trace]ID[0] expected a struct type, but found type "double"
  // RANK1: R[1][Info]ID[0] MPI_Recv at 0x{{.*}}: checking recv-buffer 0x{{.*}} of type "double" against MPI type "test_type"
  // RANK1: R[1][Trace]ID[0] expected a struct type, but found type "double"
  // clang-format on
  run_test(arr, 2, counts, offsets, types);

  // clang-format off
  // RANK0: R[0][Info]ID[1] MPI_Send at 0x{{.*}}: checking send-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK0: R[0][Trace]ID[1] expected 2 members, but the type "struct.S1" has 3 members
  // RANK1: R[1][Info]ID[1] MPI_Recv at 0x{{.*}}: checking recv-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK1: R[1][Trace]ID[1] expected 2 members, but the type "struct.S1" has 3 members
  // CHECK: R[{{0|1}}][Trace]ID[{{[0-9]+}}] expected a struct type, but found type "double"
  // clang-format on
  run_test(&s1, 2, counts, offsets, types);

  // 2: Check wrong offsets
  // clang-format off
  // RANK0: R[0][Info]ID[2] MPI_Send at 0x{{.*}}: checking send-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK0: R[0][Trace]ID[2] expected a byte offset of 24 for member 2, but the type "struct.S1" has an offset of 16
  // RANK1: R[1][Info]ID[2] MPI_Recv at 0x{{.*}}: checking recv-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK1: R[1][Trace]ID[2] expected a byte offset of 24 for member 2, but the type "struct.S1" has an offset of 16
  // CHECK: R[{{0|1}}][Trace]ID[{{[0-9]+}}] expected a struct type, but found type "double"
  // clang-format on
  run_test(&s1, 3, counts, (MPI_Aint[3]){offsetof(S1, a), offsetof(S1, c), offsetof(S1, c)}, types);

  // 3: Check wrong types
  // clang-format off
  // RANK0: R[0][Info]ID[3] MPI_Send at 0x{{.*}}: checking send-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK0: R[0][Trace]ID[3] expected a type matching MPI type "MPI_INT", but found type "double"
  // RANK0: R[0][Trace]ID[3] the typechek for member 3 failed
  // RANK1: R[1][Info]ID[3] MPI_Recv at 0x{{.*}}: checking recv-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK1: R[1][Trace]ID[3] expected a type matching MPI type "MPI_INT", but found type "double"
  // RANK1: R[1][Trace]ID[3] the typechek for member 3 failed
  // CHECK: R[{{0|1}}][Trace]ID[{{[0-9]+}}] expected a struct type, but found type "double"
  // clang-format on
  run_test(&s1, 3, counts, offsets, (MPI_Datatype[3]){MPI_DOUBLE, MPI_INT, MPI_INT});

  // 3: Check member count
  // clang-format off
  // RANK0: R[0][Info]ID[4] MPI_Send at 0x{{.*}}: checking send-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK0: R[0][Trace]ID[4] expected element count of 2 for member 1, but the type "struct.S1" has a count of 1
  // RANK1: R[1][Info]ID[4] MPI_Recv at 0x{{.*}}: checking recv-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK1: R[1][Trace]ID[4] expected element count of 2 for member 1, but the type "struct.S1" has a count of 1
  // CHECK: R[{{0|1}}][Trace]ID[{{[0-9]+}}] expected a struct type, but found type "double"
  // clang-format on
  run_test(&s1, 3, (int[3]){1, 1, 1}, offsets, types);

  // 3: Check member count
  // clang-format off
  // RANK0: R[0][Info]ID[5] MPI_Send at 0x{{.*}}: checking send-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK1: R[1][Info]ID[5] MPI_Recv at 0x{{.*}}: checking recv-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  run_test(&s1, 3, counts, offsets, types);

  // 4: Check member count with complex MPI type
  // clang-format off
  // RANK0: R[0][Info]ID[6] MPI_Send at 0x{{.*}}: checking send-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // RANK1: R[1][Info]ID[6] MPI_Recv at 0x{{.*}}: checking recv-buffer 0x{{.*}} of type "struct.S1" against MPI type "test_type"
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  MPI_Datatype member_a;
  MPI_Type_contiguous(2, MPI_DOUBLE, &member_a);
  run_test(&s1, 3, (int[3]){1, 1, 1}, offsets, (MPI_Datatype[3]){member_a, MPI_INT, MPI_DOUBLE});
  MPI_Type_free(&member_a);

  // RANK0: R[0][Info] CCounter { Send: 7 Recv: 0 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // RANK1: R[1][Info] CCounter { Send: 0 Recv: 7 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // CHECK: R[{{0|1}}][Info] MCounter { Error: 0 Null_Buf: 0 Null_Count: 0 Type_Error: 5 }
  MPI_Finalize();
  return 0;
}
