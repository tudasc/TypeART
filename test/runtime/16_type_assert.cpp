// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-typeart-alloca" %rtpath 2>&1 | FileCheck %s

#include <stdlib.h>
#include "RuntimeInterface.h"

template <typename T>
void make_assert() {
    T x[2];
    ASSERT_TYPE(&x[0], T);
}

template <typename T>
void make_assert_malloc() {
    T* x = (T*) malloc(sizeof(T));
    ASSERT_TYPE(x, T);
    free(x);
}

template <typename T1, typename T2>
void make_assert_with_type() {
    T1 x;
    ASSERT_TYPE(&x, T2);
}

struct S1 {
    int x;
};

struct S2 {
    int x;
};

typedef S1 S3;

void test_alloca();
void test_malloc();
void test_wrong_type();

int main(int argc, char** argv) {
    // CHECK: [Trace] TypeART Runtime Trace

    test_alloca();
    test_malloc();
    //test_wrong_type();

    return 0;
}

void test_alloca() {
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<char>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<short>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<int>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<long>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<float>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<double>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<int*>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert<S1>();
}

void test_malloc() {
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<char>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<short>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<int>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<long>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<float>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<double>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<int*>();
    // CHECK-NOT: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT: Assert failed
    make_assert_malloc<S1>();
}

void test_wrong_type() {
    // CHECK-NOT$: Unresolved call to __typeart_assert_type_stub
    // CHECK$: Assert failed
    make_assert_with_type<int, char>();
    // CHECK-NOT$: Unresolved call to __typeart_assert_type_stub
    // CHECK$: Assert failed
    make_assert_with_type<int*, void*>();
    // CHECK-NOT$: Unresolved call to __typeart_assert_type_stub
    // CHECK$: Assert failed
    make_assert_with_type<float, double>();
    // CHECK-NOT$: Unresolved call to __typeart_assert_type_stub
    // CHECK$: Assert failed
    make_assert_with_type<S1, int>();
    // CHECK-NOT$: Unresolved call to __typeart_assert_type_stub
    // CHECK$: Assert failed
    make_assert_with_type<S1, S1*>();
    // CHECK-NOT$: Unresolved call to __typeart_assert_type_stub
    // CHECK$: Assert failed
    make_assert_with_type<S1, S2>();
    // CHECK-NOT$: Unresolved call to __typeart_assert_type_stub
    // CHECK-NOT$: Assert failed
    make_assert_with_type<S1, S3>();

}