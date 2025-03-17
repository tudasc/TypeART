// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

// REQUIRES: dimeta

#include <cstddef>

class Y {
 public:
  std::nullptr_t null_pointer;
  void* void_pointer;
  int* other_pointer;
};

void foo() {
  Y class_y;
}

// CHECK: name:            _ZTS1Y
// CHECK-NEXT: extent:          24
// CHECK-NEXT: member_count:    3
// CHECK-NEXT: offsets:         [ 0, 8, 16 ]
// CHECK-NEXT: types:           [ 4, 3, 1 ]
// CHECK-NEXT: sizes:           [ 1, 1, 1 ]

//         0 | class Y
//         0 |   std::nullptr_t null_pointer
//         8 |   void * void_pointer
//        16 |   int * other_pointer
//           | [sizeof=24, dsize=24, align=8,
//           |  nvsize=24, nvalign=8]
