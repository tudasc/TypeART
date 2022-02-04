// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// REQUIRES: openmp
// clang-format on
extern void MPI_call(void*);

void func(int* x) {
#pragma omp parallel
  { MPI_call(x); }
}

void foo() {
  // check-inst: define {{.*}} @foo
  // check-inst: %x = alloca
  // check-inst: %0 = bitcast i32* %x to i8*
  // check-inst: call void @__typeart_alloc_stack(i8* %0, i32 2, i64 1)
  // check-inst-not: __typeart_alloc_stack_omp
  int x;
#pragma omp parallel
  { func(&x); }
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   1
// CHECK-NEXT: Global :   0