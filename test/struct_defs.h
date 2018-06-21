#ifndef STRUCT_DEFS_H
#define STRUCT_DEFS_H

// Some struct definitions to be used in runtime and type mapping tests
// The number behind each member indicates the corresponding offset.
// The comment after the struct definition gives the extent with added padding.

// Note: The given IDs are only valid, if no other structs/classes are previously defined

// ID: 11
#define S_INT_ID 11
typedef struct s_int_t {
    int a; // 0
} s_int; // 4

// ID: 12
#define S_BUILTINS_ID 12
typedef struct s_builtins_t {
    int a;   // 0
    char b;  // 4
    long c;  // 8
} s_builtins; // 16

// ID: 13
#define S_ARRAYS_ID 13
typedef struct s_arrays_t {
    int a[3];           // 0
    long b[2];          // 16
    char c;             // 32
    int d[3];  // 36
    char e[5];          // 48
    long f;    // 56
    char g;
} s_arrays; // 72

// ID: 14
#define S_PTRS_ID 14
typedef struct s_ptrs_t {
    char a; // 0
    int* b; // 8
    int c; // 16
    double* d; // 20
} s_ptrs; // 32

// ID: 15
#define S_MIXED_SIMPLE_ID 15
typedef struct s_mixed_simple_t {
    int a;           // 0
    double b[3];     // 8
    char* c;     // 32
    char d[5];  // 40
} s_mixed_simple;  // 48

// ID: 16
#define S_PTR_TO_SELF_ID 16
typedef struct s_ptr_to_self_t {
    struct s_ptr_to_self_t* a;  // 0
    struct s_ptr_to_self_t** b; // 8
} s_ptr_to_self;  // 16

// ID: 17
#define S_STRUCT_MEMBER_ID 17
typedef struct s_struct_member_t {
    int a; // 0
    s_ptr_to_self b;  // 4
    struct s_struct_member_t* c; // 20
} s_struct_member; // 32

#define S_AOS_ID 18
typedef struct s_aos_t {
    int a; // 0
    s_struct_member b[2]; // 8
    s_struct_member* c[3]; // 72
} s_aos; // 96

#endif