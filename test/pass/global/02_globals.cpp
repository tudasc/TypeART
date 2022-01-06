// RUN: %cpp-to-llvm %s | %apply-typeart -typeart-global -S 2>&1 | %filecheck %s

int global;
int global_2 = 0;
extern int global_3;
extern int global_4;
static int global_5;
static int global_6 = 0;

extern void bar(int*);

void foo() {
  bar(&global);
  bar(&global_2);
  bar(&global_3);
  bar(&global_5);
  bar(&global_6);
}

// CHECK: void @__typeart_init_module_
// CHECK-NEXT: entry:
// CHECK-DAG: call void @__typeart_alloc_global(i8* bitcast (i32* @global to i8*)
// CHECK-DAG: call void @__typeart_alloc_global(i8* bitcast (i32* @global_2 to i8*)
// CHECK-DAG: call void @__typeart_alloc_global(i8* bitcast (i32* @{{.*}}global_5 to i8*)
// CHECK-DAG: call void @__typeart_alloc_global(i8* bitcast (i32* @{{.*}}global_6 to i8*)
// CHECK: ret void
