// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-filter-pointer-alloca=false -typeart-stack-lifetime=false -S 2>&1 | %filecheck %s
// clang-format on
void test() {
  int a[100];
  int* pa = a;  // TODO: Tracking this value should not be necessary?
}

// CHECK: @test()
// CHECK: %__ta_alloca_counter = alloca i32
// CHECK-NEXT: store i32 0, i32* %__ta_alloca_counter

// CHECK: [[POINTER:%[0-9a-z]+]] = alloca [100 x i32]
// CHECK-NEXT: [[POINTER2:%[0-9a-z]+]] = bitcast [100 x i32]* [[POINTER]] to i8*
// CHECK-NEXT: call void @__typeart_alloc_stack(i8* [[POINTER2]], i32 2, i64 100)

// CHECK: call void @__typeart_leave_scope(i32 %__ta_counter_load)

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}2
