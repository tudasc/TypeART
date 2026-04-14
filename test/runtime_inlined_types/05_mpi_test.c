// RUN: %wrapper-mpicc -g %s -o %s.exe
// RUN: %mpi-exec -np 1 %s.exe 2>&1 | %filecheck %s

// REQUIRES: mpicc && !llvm-14

#include <mpi.h>
void MPI_mock(MPI_Datatype*) {
}

int main(void) {
  MPI_Datatype array_of_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
  MPI_mock(&array_of_types[0]);
  return 0;
}

// CHECK-NOT: Error
