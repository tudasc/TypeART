// clang-format off
// RUN: rm %type_file | %c-to-llvm %s | %apply-typeart -S 2>&1; cat %type_file | %filecheck %s
// clang-format on

// Note: This test assumes standard alignment on a 64bit system. Non-standard alignment may lead to failure.

// typedef enum typeart_builtin_type_t {  // NOLINT
//    TA_INT8 = 0,
//    TA_INT16 = 1,
//    TA_INT32 = 2,
//    TA_INT64 = 3,
//
//    // Note: Unsigned types are currently not supported
//    // TA_UINT8,
//    // TA_UINT16,
//    // TA_UINT32,
//    // TA_UINT64,
//
//    TA_HALF = 4,       // IEEE 754 half precision floating point type
//    TA_FLOAT = 5,      // IEEE 754 single precision floating point type
//    TA_DOUBLE = 6,     // IEEE 754 double precision floating point type
//    TA_FP128 = 7,      // IEEE 754 quadruple precision floating point type
//    TA_X86_FP80 = 8,   // x86 extended precision 80-bit floating point type
//    TA_PPC_FP128 = 9,  // ICM extended precision 128-bit floating point type
//
//    TA_PTR = 10, // Represents all pointer types
//
//    TA_NUM_VALID_IDS = TA_PTR + 1,
//
//    TA_UNKNOWN_TYPE = 255,
//    TA_NUM_RESERVED_IDS = TA_UNKNOWN_TYPE + 1
//} typeart_builtin_type;

#include <stdlib.h>

// TODO: Use structs from struct_defs.h

typedef struct s_t {
  int a;
} s;

typedef struct s2_t {
  int a;   // 0
  char b;  // 4
  long c;  // 8
} s2;

typedef struct s3_t {
  int a[3];           // 0
  long b[2];          // 16
  char c;             // 32
  unsigned int d[3];  // 36
  char e[5];          // 48
  unsigned long f;    // 56
} s3;

typedef struct s4_t {
  int a;           // 0
  double b[3];     // 8
  double c[3];     // 32
  struct s4_t* d;  // 56
} s4;

int main(int argc, char** argv) {
  s* a  = malloc(sizeof(s));
  s2* b = malloc(sizeof(s2));
  s3* c = malloc(sizeof(s3));
  s4* d = malloc(sizeof(s4));
  free(d);
  free(c);
  free(b);
  free(a);
  return 0;
}

// CHECK: - id:              256
// CHECK: name:            struct.s_t
// CHECK: extent:          4
// CHECK: member_count:    1
// CHECK: offsets:         [ 0 ]
// CHECK: types:           [ 2 ]
// CHECK: sizes:           [ 1 ]

// CHECK: - id:              257
// CHECK: name:            struct.s2_t
// CHECK:         extent:          16
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 4, 8 ]
// CHECK: types:           [ 2, 0, 3 ]
// CHECK: sizes:           [ 1, 1, 1 ]

// CHECK: - id:              258
// CHECK: name:            struct.s3_t
// CHECK:         extent:          64
// CHECK: member_count:    6
// CHECK: offsets:         [ 0, 16, 32, 36, 48, 56 ]
// CHECK: types:           [ 2, 3, 0, 2, 0, 3 ]
// CHECK: sizes:           [ 3, 2, 1, 3, 5, 1 ]

// CHECK: - id:              259
// CHECK: name:            struct.s4_t
// CHECK:         extent:          64
// CHECK: member_count:    4
// CHECK: offsets:         [ 0, 8, 32, 56 ]
// CHECK: types:           [ 2, 6, 6, 10 ]
// CHECK: sizes:           [ 1, 3, 3, 1 ]
