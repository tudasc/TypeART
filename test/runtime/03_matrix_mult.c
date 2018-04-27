// RUN: %scriptpath/applyAndRun.sh %s %pluginname -must %pluginpath %rtpath | FileCheck %s

#include <stdlib.h>
#include <string.h>

typedef struct mat_t {
  double* vals;
  int dim[2];
} mat;

mat alloc_mat(int rows, int cols) {
  mat m;
  m.dim[0] = rows;
  m.dim[1] = cols;
  m.vals = (double*)malloc(rows * cols * sizeof(double));
  return m;
}

void free_mat(mat m) {
  free(m.vals);
}

void fill(mat m, double val) {
  for (int i = 0; i < m.dim[0] * m.dim[1]; i++) {
    m.vals[i] = val;
  }
}

int multiply(mat a, mat b, mat result) {
  int rows = a.dim[0];
  int cols = b.dim[1];

  int n = a.dim[1];

  if (n != b.dim[0] || result.dim[0] != rows || result.dim[1] != cols)
    return 0;

  int num_vals = rows * cols;

  double* temp = malloc(num_vals * sizeof(double));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double val = 0;
      for (int k = 0; k < n; k++) {
        val += a.vals[i * cols + k] * b.vals[k * cols + j];
      }
      temp[i * cols + j] = val;
    }
  }

  memcpy(result.vals, temp, num_vals);
  free(temp);

  return 1;
}

void test_multiply() {
  const int n = 8;
  const int k = 4;

  mat matrices[k];
  for (int i = 0; i < k; i++) {
    matrices[i] = alloc_mat(n, n);
    fill(matrices[i], i + 1);
  }

  for (int i = 1; i < k; i++) {
    multiply(matrices[0], matrices[i], matrices[0]);
  }

  for (int i = 0; i < k; i++) {
    free_mat(matrices[i]);
  }
}

int main(int argc, char** argv) {
  test_multiply();
  return 0;
}

// CHECK: MUST Support Runtime Trace

// Alloc matrix array
// CHECK: Alloc    0x{{.*}}   struct.mat_t   16     4

// Alloc matrix values
// CHECK: Alloc    0x{{.*}}   float64         8    64
// CHECK: Alloc    0x{{.*}}   float64         8    64
// CHECK: Alloc    0x{{.*}}   float64         8    64
// CHECK: Alloc    0x{{.*}}   float64         8    64

// Alloc and free temp buffer
// CHECK: Alloc    0x{{.*}}   float64         8    64
// CHECK: Free     0x{{.*}}
// CHECK: Alloc    0x{{.*}}   float64         8    64
// CHECK: Free     0x{{.*}}
// CHECK: Alloc    0x{{.*}}   float64         8    64
// CHECK: Free     0x{{.*}}

// Free matrix values
// CHECK: Free     0x{{.*}}
// CHECK: Free     0x{{.*}}
// CHECK: Free     0x{{.*}}
// CHECK: Free     0x{{.*}}

// TODO: Handle stack deallocation?
// Free matrix array
// CHECK?: Free     0x{{.*}}
