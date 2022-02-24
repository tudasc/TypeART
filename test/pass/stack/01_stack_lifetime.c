// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-stack-lifetime -S 2>&1 | %filecheck %s
// clang-format on

extern void type_check(void*);

void correct(int rank) {
  if (rank == 1) {
    int buffer[3][3] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    type_check(buffer);
    type_check(&buffer[2][2]);
  } else {
    int rcv[3] = {0, 1, 2};
    type_check(rcv);
  }
}

// CHECK: [[POINTER:%[0-9a-z]+]] = bitcast [3 x [3 x i32]]* [[BUF:%[0-9a-z]+]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 36, i8* [[POINTER]])
// CHECK-NEXT: call void @__typeart_alloc_stack(i8* [[POINTER]], i32 2, i64 9)

// CHECK: call void @llvm.lifetime.start.p0i8(i64 12, i8* [[POINTER2:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_alloc_stack(i8* [[POINTER2]], i32 2, i64 3)
