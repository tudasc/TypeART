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

typedef struct s1_t {
  char a[3];       // 0
  struct s1_t* b;  // 8
} s1;

typedef struct s2_t {
  s1 a;            // 0
  s1* b;           // 16
  struct s2_t* c;  // 24
} s2;

typedef struct s3_t {
  s1 a[2];   // 0
  char b;    // 32
  s2* c[3];  // 40
} s3;

int main(int argc, char** argv) {
  s1* a = malloc(sizeof(s1));
  s2* b = malloc(sizeof(s2));
  s3* c = malloc(sizeof(s3));
  free(c);
  free(b);
  free(a);
  return 0;
}

// CHECK: - id:              256
// CHECK: name:            struct.s1_t
// CHECK:         extent:          16
// CHECK: member_count:    2
// CHECK: offsets:         [ 0, 8 ]
// CHECK: types:           [ 0, 10 ]
// CHECK: sizes:           [ 3, 1 ]

// CHECK: - id:              257
// CHECK: name:            struct.s2_t
// CHECK:         extent:          32
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 16, 24 ]
// CHECK: types:           [ 256, 10, 10 ]
// CHECK: sizes:           [ 1, 1, 1 ]

// CHECK: - id:              258
// CHECK: name:            struct.s3_t
// CHECK:         extent:          64
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 32, 40 ]
// CHECK: types:           [ 256, 0, 10 ]
// CHECK: sizes:           [ 2, 1, 3 ]
