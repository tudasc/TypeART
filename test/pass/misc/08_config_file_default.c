// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// CHECK-NOT: {{(Error|Fatal)}}

// CHECK: types:           {{.*}}.yaml
// CHECK-NEXT: heap:            true
// CHECK-NEXT: stack:           false
// CHECK-NEXT: global:          false
// CHECK-NEXT: stats:           true
// CHECK-NEXT: stack-lifetime:  true
// CHECK-NEXT: typegen:         {{dimeta|ir}}
// CHECK-NEXT: filter:          false
// CHECK-NEXT: call-filter:
// CHECK-NEXT:   implementation:  std
// CHECK-NEXT:   glob:            '*MPI_*'
// CHECK-NEXT:   glob-deep:       'MPI_*'
// CHECK-NEXT:   cg-file:         ''
// CHECK-NEXT: analysis:
// CHECK-NEXT:   filter-global:   true
// CHECK-NEXT:   filter-heap-alloca: false
// CHECK-NEXT:   filter-pointer-alloca: true
// CHECK-NEXT:   filter-non-array-alloca: false

void test() {
}
