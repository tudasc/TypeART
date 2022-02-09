// clang-format off
// RUN: %run %s --optimization -O2 --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | %filecheck %s
// RUN: cat "%s.log/1/rank.1/stderr" | %filecheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

// XFAIL: *

#include <mpi.h>

typedef struct {
  float imag;
  float real;
} complex;

void g_complexsum(complex* cpt) {
  complex work;
  MPI_Allreduce(cpt, &work, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  *cpt = work;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // CHECK-NOT: R[{{0|1}}][Error]{{.*}}
  complex c{1.0, 2.0};
  g_complexsum(&c);

  MPI_Finalize();
  return 0;
}
