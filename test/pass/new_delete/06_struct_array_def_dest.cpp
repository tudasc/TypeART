// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

struct S1 {
  int x;
  ~S1() = default;
};

// CHECK: call{{.*}} {{i8\*|ptr}} @_Znam(i64{{( noundef)?}} 8)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 2)
int main() {
  S1* ss = new S1[2];
  return 0;
}
