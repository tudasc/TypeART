// RUN: %c-to-llvm -fprofile-instr-generate -fcoverage-mapping %s | %apply-typeart -typeart-stack -S 2>&1 \
// RUN: | %filecheck %s

int global;
int global_2 = 0;

extern void bar(int*);

void foo() {
  bar(&(&global)[3]);
  bar(&global_2);
}

// CHECK: void @__typeart_init_module_
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__typeart_alloc_global(i8* bitcast (i32* @global_2 to i8*)
// CHECK-NEXT: call void @__typeart_alloc_global(i8* bitcast (i32* @global to i8*)
// CHECK-NEXT: ret void
