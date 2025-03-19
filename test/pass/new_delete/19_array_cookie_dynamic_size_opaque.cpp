// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// REQUIRES: llvm-18 || llvm-19
// clang-format on

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

struct S1 {
  int x;
  ~S1(){};
};

// CHECK: [[MEM:%[0-9a-z]+]] = call{{.*}} ptr @_Znam(i64{{( noundef)?}} [[ALLOC:%[0-9a-z]+]])
// CHECK: store i64 [[COUNT:%[0-9a-z]+]], ptr [[MEM]], align 8
// CHECK: [[ARR:%[0-9a-z]+]] = getelementptr inbounds i8, ptr [[MEM]], i64 8
// CHECK: call void @__typeart_alloc(ptr [[ARR]], i32 {{2[0-9]+}}, i64 [[COUNT]])
int main() {
  volatile int elment_count = 2;
  S1* ss                    = new S1[elment_count];
  return 0;
}
