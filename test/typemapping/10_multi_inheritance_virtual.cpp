// RUN: %remove %tu_yaml
// RUN: %cpp-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

// REQUIRES: dimeta

class Base {
 public:
  double x;

  virtual void foo(){};
};

class X {
 public:
  int y;

  virtual int bar() {
    return y;
  };
};

class Y : public X, public Base {
 public:
  float z;
};

void foo() {
  Y class_y;
}

// CHECK: name:            _ZTS1X
// CHECK-NEXT: extent:          16
// CHECK-NEXT: member_count:    2
// CHECK-NEXT: offsets:         [ 0, 8 ]
// CHECK-NEXT: types:           [ 2, 13 ]
// CHECK-NEXT: sizes:           [ 1, 1 ]
// CHECK-NEXT: flags:           1

// CHECK: name:            _ZTS1Y
// CHECK-NEXT: extent:          40
// CHECK-NEXT: member_count:    3
// CHECK-NEXT: offsets:         [ 0, 16, 32 ]
// CHECK-NEXT: types:           [ 25{{[6-9]}}, 25{{[6-9]}}, 23 ]
// CHECK-NEXT: sizes:           [ 1, 1, 1 ]
// CHECK-NEXT: flags:           1

//          0 | class Y
//          0 |   class X (primary base)
//          0 |     (X vtable pointer)
//          8 |     int y
//         16 |   class Base (base)
//         16 |     (Base vtable pointer)
//         24 |     double x
//         32 |   float z
//            | [sizeof=40, dsize=36, align=8,
//            |  nvsize=36, nvalign=8]
