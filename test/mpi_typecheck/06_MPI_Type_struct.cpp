// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan
// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | %filecheck --check-prefixes CHECK,RANK0 %s
// RUN: cat "%s.log/1/rank.1/stderr" | %filecheck --check-prefixes CHECK,RANK1 %s
// clang-format on

// XFAIL: *

#include <mpi.h>

struct S1 {
  double a[2];
  int b;
  double c;
};

struct S2 {
  S1 s1;
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
  S2 s2;

  // 1: Check non-struct buffer type and wrong member count
  // clang-format off
  // RANK0: R[0][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: type error while checking send-buffer 0x{{.*}} of type [3 x double] against 1 element of MPI type "test_type": expected a struct type, but found type "double"
  // RANK1: R[1][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: type error while checking recv-buffer 0x{{.*}} of type [3 x double] against 1 element of MPI type "test_type": expected a struct type, but found type "double"
  // clang-format on
  run_test(arr, 2, counts, offsets, types);

  // clang-format off
  // RANK0: R[0][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: type error while checking send-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": expected 2 members, but the type "struct.S1" has 3 members. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // RANK1: R[1][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: type error while checking recv-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": expected 2 members, but the type "struct.S1" has 3 members. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // clang-format on
  run_test(&s1, 2, counts, offsets, types);

  // 2: Check wrong offsets
  // clang-format off
  // RANK0: R[0][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: type error while checking send-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": expected a byte offset of 24 for member 2, but the type "struct.S1" has an offset of 16. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // RANK1: R[1][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: type error while checking recv-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": expected a byte offset of 24 for member 2, but the type "struct.S1" has an offset of 16. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // clang-format on
  run_test(&s1, 3, counts, (MPI_Aint[3]){offsetof(S1, a), offsetof(S1, c), offsetof(S1, c)}, types);

  // 3: Check wrong types
  // clang-format off
  // RANK0: R[0][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: type error while checking send-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": the typecheck for member 3 failed (expected a type matching MPI type "MPI_INT", but found type "double"). Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // RANK1: R[1][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: type error while checking recv-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": the typecheck for member 3 failed (expected a type matching MPI type "MPI_INT", but found type "double"). Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // clang-format on
  run_test(&s1, 3, counts, offsets, (MPI_Datatype[3]){MPI_DOUBLE, MPI_INT, MPI_INT});

  // 3: Check member count
  // clang-format off
  // RANK0: R[0][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: type error while checking send-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": expected element count of 2 for member 1, but the type "struct.S1" has a count of 1. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // RANK1: [1][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: type error while checking recv-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type": expected element count of 2 for member 1, but the type "struct.S1" has a count of 1. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // clang-format on
  run_test(&s1, 3, (int[3]){1, 1, 1}, offsets, types);

  // 4: Check member count
  // clang-format off
  // RANK0: R[0][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: successfully checked send-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type"
  // RANK1: R[1][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: successfully checked recv-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type"
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  run_test(&s1, 3, counts, offsets, types);

  // 5: Check member count with complex MPI type
  // clang-format off
  // RANK0: R[0][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: successfully checked send-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type"
  // RANK1: R[1][Info]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: successfully checked recv-buffer 0x{{.*}} of type [1 x struct.S1] against 1 element of MPI type "test_type"
  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  // clang-format on
  MPI_Datatype member_a;
  MPI_Type_contiguous(2, MPI_DOUBLE, &member_a);
  run_test(&s1, 3, (int[3]){1, 1, 1}, offsets, (MPI_Datatype[3]){member_a, MPI_INT, MPI_DOUBLE});
  MPI_Type_free(&member_a);

  // 6: Check error output for multiple recursions
  // clang-format off
  // RANK0: R[0][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: type error while checking send-buffer 0x{{.*}} of type [1 x struct.S2] against 1 element of MPI type "test_type": expected 3 members, but the type "struct.S2" has 1 members. Tried the first member [1 x struct.S1] of struct type "struct.S2" with error: expected element count of 2 for member 1, but the type "struct.S1" has a count of 1. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // RANK1: R[1][Error]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Recv: type error while checking recv-buffer 0x{{.*}} of type [1 x struct.S2] against 1 element of MPI type "test_type": expected 3 members, but the type "struct.S2" has 1 members. Tried the first member [1 x struct.S1] of struct type "struct.S2" with error: expected element count of 2 for member 1, but the type "struct.S1" has a count of 1. Tried the first member [2 x double] of struct type "struct.S1" with error: expected a struct type, but found type "double" ]
  // clang-format on
  run_test(&s2, 3, (int[3]){1, 1, 1}, offsets, types);

  // RANK0: R[0][Info]T[{{[0-9]*}}] CCounter { Send: 8 Recv: 0 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // RANK1: R[1][Info]T[{{[0-9]*}}] CCounter { Send: 0 Recv: 8 Send_Recv: 0 Unsupported: 0 MAX RSS[KBytes]: {{[0-9]+}} }
  // CHECK: R[{{0|1}}][Info]T[{{[0-9]*}}] MCounter { Error: 0 Null_Buf: 0 Null_Count: 0 Type_Error: 6 }
  MPI_Finalize();
  return 0;
}
