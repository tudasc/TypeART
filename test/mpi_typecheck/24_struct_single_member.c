// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 %s.exe" 2>&1 | %filecheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

struct VecT {
  double a;
  double b;
};

void run_struct_vec(int rank) {
  // CHECK: MPI_Send: successfully checked send-buffer
  if (rank == 0) {
    MPI_Datatype vec_struct_ty;
    int length[1]         = {1};
    MPI_Aint offsets[1]   = {offsetof(struct VecT, b)};
    MPI_Datatype types[1] = {MPI_DOUBLE};
    MPI_Type_create_struct(1, length, offsets, types, &vec_struct_ty);
    MPI_Type_commit(&vec_struct_ty);

    struct VecT buffer[1] = {1., 2.};
    MPI_Send(&buffer[0], 1, vec_struct_ty, 1, 0, MPI_COMM_WORLD);

  } else {
    double received[1];
    MPI_Recv(&received, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    fprintf(stderr, "[Test] Received: %.1f\n", received[0]);
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  run_struct_vec(rank);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
