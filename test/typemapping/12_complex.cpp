// RUN: %remove %tu_yaml
// RUN: %cpp-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

// REQUIRES: dimeta

#include <complex>

class Y {
 public:
  std::complex<float> f_cplx;
  std::complex<double> d_cplx;
  std::complex<long double> ld_cplx;
};

void foo() {
  Y class_y;
}

// CHECK:   name:            _ZTSSt7complexIfE
// CHECK-NEXT:   extent:          8
// CHECK-NEXT:   member_count:    1
// CHECK-NEXT:   offsets:         [ 0 ]
// CHECK-NEXT:   types:           [ 26 ]
// CHECK-NEXT:   sizes:           [ 1 ]

// CHECK:   name:            _ZTSSt7complexIdE
// CHECK-NEXT:   extent:          16
// CHECK-NEXT:   member_count:    1
// CHECK-NEXT:   offsets:         [ 0 ]
// CHECK-NEXT:   types:           [ 27 ]
// CHECK-NEXT:   sizes:           [ 1 ]

// CHECK:   name:            _ZTSSt7complexIeE
// CHECK-NEXT:   extent:          32
// CHECK-NEXT:   member_count:    1
// CHECK-NEXT:   offsets:         [ 0 ]
// CHECK-NEXT:   types:           [ 28 ]
// CHECK-NEXT:   sizes:           [ 1 ]

// CHECK: name:            _ZTS1Y
// CHECK-NEXT: extent:          64
// CHECK-NEXT: member_count:    3
// CHECK-NEXT: offsets:         [ 0, 8, 32 ]
// CHECK-NEXT: types:           [ 25{{[0-9]}}, 25{{[0-9]}}, 25{{[0-9]}} ]
// CHECK-NEXT: sizes:           [ 1, 1, 1 ]

//         0 | class Y
//         0 |   class std::complex<float> f_cplx
//         0 |     _ComplexT _M_value
//         8 |   class std::complex<double> d_cplx
//         8 |     _ComplexT _M_value
//        32 |   class std::complex<long double> ld_cplx
//        32 |     _ComplexT _M_value
//           | [sizeof=64, dsize=64, align=16,
//           |  nvsize=64, nvalign=16]
