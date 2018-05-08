// RUN: %scriptpath/applyAndRun.sh %s %pluginname -must %pluginpath %rtpath | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../runtime/RuntimeInterface.h"
//#include "../../runtime/CRuntime.h"

//int mustCheckType(void* addr, int typeId);
//int mustCheckTypeName(void* addr, const char* typeName);

typedef struct vector_t {
  double* vals;
  int size;
} vector;

vector alloc_vector(int n) {
  vector v;
  v.size = n;
  v.vals = malloc(n * sizeof(double));
  return v;
}

void free_vector(vector v) {
  free(v.vals);
}

int fill_vector(void* values, int count, vector* v) {
  must_builtin_type type;
  lookup_result result = must_support_get_builtin_type(values, &type);
  if (result == SUCCESS && type == C_DOUBLE) {
    memcpy(v->vals, values, count);
    v->size = count;
    printf("Success\n");
    return 1;
  }
  printf("Failure\n");
  return 0;
}

int main(int argc, char** argv) {
  const int n = 3;
  // CHECK: Alloc    0x{{.*}}   int   4     3
  int int_vals[3] = {1, 2, 3};
  // CHECK: Alloc    0x{{.*}}   double   8     3
  double d_vals[3] = {1, 2, 3};
  // CHECK: Alloc    0x{{.*}}   float   4     3
  float f_vals[3] = {1, 2, 3};
  // CHECK: Alloc    0x{{.*}}   double   8     3
  vector v = alloc_vector(n);
  // CHECK: Alloc    0x{{.*}}   double   8     3
  vector w = alloc_vector(n);
  // CHECK: Success
  fill_vector(w.vals, n, &v);
  // CHECK: Failure
  fill_vector(int_vals, n, &v);
  // CHECK: Success
  fill_vector(d_vals, n, &v);
  // CHECK: Failure
  fill_vector(f_vals, n, &v);
  // CHECK: Free     0x{{.*}}
  free_vector(w);
  // CHECK: Free     0x{{.*}}
  free_vector(v);
  return 0;
}
