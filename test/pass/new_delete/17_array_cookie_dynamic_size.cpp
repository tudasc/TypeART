// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

struct S1 {
  int x;
  ~S1(){};
};

// CHECK: [[MEM:%[0-9a-z]+]] = call{{.*}} i8* @_Znam(i64{{( noundef)?}} [[ALLOC:%[0-9a-z]+]])
// CHECK: [[COOKIE:%[0-9a-z]+]] = bitcast i8* [[MEM]] to i64*
// CHECK: store i64 [[COUNT:%[0-9a-z]+]], i64* [[COOKIE]], align 8
// CHECK: [[ARR:%[0-9a-z]+]] = getelementptr inbounds i8, i8* [[MEM]], i64 8
// CHECK: call void @__typeart_alloc(i8* [[ARR]], i32 {{2[0-9]+}}, i64 [[COUNT]])
// CHECK: bitcast i8* [[ARR]] to %struct.S1*
int main() {
  volatile int elment_count = 2;
  S1* ss                    = new S1[elment_count];
  return 0;
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
