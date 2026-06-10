// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s --check-prefixes CHECK,%llvm-version-check
// clang-format on

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

struct S1 {
  int x;
  ~S1() {};
};

// LLVM_LEGACY: [[MEM:%[0-9a-z]+]] = call{{.*}} i8* @_Znam(i64{{( noundef)?}} [[ALLOC:%[0-9a-z]+]])
// LLVM_LEGACY: [[COOKIE:%[0-9a-z]+]] = bitcast i8* [[MEM]] to i64*
// LLVM_LEGACY: store i64 [[COUNT:%[0-9a-z]+]], i64* [[COOKIE]], align 8
// LLVM_LEGACY: [[ARR:%[0-9a-z]+]] = getelementptr inbounds i8, i8* [[MEM]], i64 8
// LLVM_LEGACY: call void @__typeart_alloc(i8* [[ARR]], i32 {{2[0-9]+}}, i64 [[COUNT]])
// LLVM_LEGACY: bitcast i8* [[ARR]] to %struct.S1*
// LLVM: [[MEM:%[0-9a-z]+]] = call{{.*}} ptr @_Znam(i64{{( noundef)?}} [[ALLOC:%[0-9a-z]+]])
// LLVM: store i64 [[COUNT:%[0-9a-z]+]], ptr [[MEM]], align 8
// LLVM: [[ARR:%[0-9a-z]+]] = getelementptr inbounds i8, ptr [[MEM]], i64 8
// LLVM: call void @__typeart_alloc{{(_mty)?}}(ptr [[ARR]]
int main() {
  volatile int elment_count = 2;
  S1* ss                    = new S1[elment_count];
  return 0;
}
