// RUN: %c-to-llvm --coverage %s | %apply-typeart --typeart-global=true -S 2>&1 | %filecheck %s --check-prefixes CHECK,%llvm-version-check

int global;
int global_2 = 0;

extern void bar(int*);

void foo() {
  bar(&global);
  bar(&global_2);
}

// CHECK: void @__typeart_init_module_
// CHECK-NEXT: entry:
// LLVM_LEGACY-NEXT: call void @__typeart_alloc_global(i8* bitcast ({{[^@]+}} @global_2 to i8*)
// LLVM-NEXT: call void @__typeart_alloc_global(ptr @global_2,
// LLVM_LEGACY-NEXT: call void @__typeart_alloc_global(i8* bitcast ({{[^@]+}} @global to i8*)
// LLVM-NEXT: call void @__typeart_alloc_global(ptr @global,
// LLVM_LEGACY-NEXT: ret void
// LLVM-NEXT: ret void
