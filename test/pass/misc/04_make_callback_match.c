// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -S 2>&1 | %filecheck %s

void __typeart_leave_scope(int alloca_count);

int main(void) {
  __typeart_leave_scope(0);
  return 0;
}

// CHECK:      TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   1
