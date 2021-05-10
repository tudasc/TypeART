// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// clang-format on

struct alignas(16) S1 {
  int x;
  ~S1(){};
};

// CHECK: call i8* @_Znam(i64 48)
// CHECK: call void @__typeart_alloc(i8* [[MEM:%[0-9]+]], i32 {{2[0-9]+}}, i64 2)
// CHECK: [[ARR:%[0-9]+]] = getelementptr inbounds i8, i8* [[MEM]], i64 16
// CHECK: bitcast i8* [[ARR]] to %struct.S1*
int main() {
  S1* ss = new S1[2];
  return 0;
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0