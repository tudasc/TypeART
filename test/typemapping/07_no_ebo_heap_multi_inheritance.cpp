// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

struct Base {
  static double s_mem;
  double s_mem_2;
  static double s_mem_3;
};

struct BaseTwo {
  double s_mem;
};

struct Derived : public Base, BaseTwo {
  double pad;
  int* c;
};

void foo() {
  Derived d;
}

// CHECK: name:            {{(_ZTS7Derived|struct.Derived)}}
// CHECK-NEXT: extent:          32
// CHECK-NEXT: member_count:    4
// CHECK-NEXT: offsets:         [ 0, 8, 16, 24 ]
// CHECK-NEXT: types:           [ 25{{[6-9]}}, 25{{[6-9]}}, 24, 1 ]
// CHECK-NEXT: sizes:           [ 1, 1, 1, 1 ]
// CHECK-NEXT: flags:           1

//          0 | struct Derived
//          0 |   struct Base (base)
//          0 |     double s_mem_2
//          8 |   struct BaseTwo (base)
//          8 |     double s_mem
//         16 |   double pad
//         24 |   int * c
//            | [sizeof=32, dsize=32, align=8,
//            |  nvsize=32, nvalign=8]
