// clang-format off
// RUN: %remove %tu_yaml && %c-to-llvm %s | %apply-typeart --typeart-stack=true -S 2>&1 | %filecheck %s
// clang-format on

typedef float float2 __attribute__((ext_vector_type(2)));

void test() {
  float2 vf = (float2)(1.0f, 2.0f);
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}1

// CHECK-NOT Type is not supported: <2 x float>
// CHECK: alloca <2 x float>, align 8
// CHECK: call void @__typeart_alloc_stack(i8* %{{[0-9]}}, i32 25{{[0-9]}}, i64 1)
