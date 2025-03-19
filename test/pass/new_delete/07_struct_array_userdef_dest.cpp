// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// Wrong size is calculated due to using Znam call, instead of bitcast to struct.S1*
// REQUIRES: dimeta
// clang-format on

struct S1 {
  int x;
  ~S1(){};
};

// CHECK: call{{.*}} {{i8\*|ptr}} @_Znam(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 2)
int main() {
  S1* ss = new S1[2];
  return 0;
}
