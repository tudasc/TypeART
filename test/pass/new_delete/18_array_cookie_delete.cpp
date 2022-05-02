// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

struct S1 {
  int x;
  ~S1(){};
};

// CHECK: [[MEM:%[0-9a-z]+]] = getelementptr inbounds i8, i8* [[ARR:%[0-9a-z]+]], i64 -8
// CHECK: call void @_ZdaPv(i8*{{( noundef)?}} [[MEM]])
// CHECK: call void @__typeart_free(i8* [[ARR]])
int main() {
  S1* ss = new S1[2];
  delete[] ss;
  return 0;
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
