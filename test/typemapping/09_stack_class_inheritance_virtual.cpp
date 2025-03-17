// RUN: %remove %tu_yaml
// RUN: %cpp-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

// REQUIRES: dimeta

class Base {
 public:
  double x;

  virtual double foo() {
    return x;
  }
};

class X : public Base {
 public:
  int y;
  char c;
  unsigned char d;

  double foo() {
    return y;
  }
};
int foo() {
  X class_x;
  return class_x.y;
}

// CHECK:   name:            _ZTS1X
// CHECK-NEXT:  extent:          24
// CHECK-NEXT:  member_count:    4
// CHECK-NEXT:  offsets:         [ 0, 16, 20, 21 ]
// CHECK-NEXT:  types:           [ 25{{[0-9]}}, 13, 6, 7 ]
// CHECK-NEXT:  sizes:           [ 1, 1, 1, 1 ]
// CHECK-NEXT:  flags:           1

// *** Dumping AST Record Layout
//          0 | class Base
//          0 |   (Base vtable pointer)
//          8 |   double x
//            | [sizeof=16, dsize=16, align=8,
//            |  nvsize=16, nvalign=8]
// *** Dumping AST Record Layout
//          0 | class X
//          0 |   class Base (primary base)
//          0 |     (Base vtable pointer)
//          8 |     double x
//         16 |   int y
//         20 |   char c
//         21 |   unsigned char d
//            | [sizeof=24, dsize=22, align=8,
//            |  nvsize=22, nvalign=8]
