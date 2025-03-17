// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

struct Base {
  static double s_mem;
  static double s_mem_2;
  static double s_mem_3;
};

struct BaseTwo {
  static double other_s_mem;
};

struct Derived : public Base, BaseTwo {
  double pad;
  int* c;
};

void foo() {
  Derived d;
}

// CHECK:  name:            {{(_ZTS7Derived|struct.Derived)}}
// CHECK-NEXT:  extent:          16
// CHECK-NEXT:  member_count:    2
// CHECK-NEXT:  offsets:         [ 0, 8 ]
// CHECK-NEXT:  types:           [ 24, 1 ]
// CHECK-NEXT:  sizes:           [ 1, 1 ]
