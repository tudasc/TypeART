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
  // CHECK: MPI_Send: type error
  if (rank == 0) {
    MPI_Datatype vec_struct_ty;
    int length[2]         = {1, 1};
    MPI_Aint offsets[2]   = {offsetof(struct VecT, a), offsetof(struct VecT, b)};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(1, length, offsets, types, &vec_struct_ty);

    MPI_Datatype hvec_type;
    // Error with stride and buffer[3] (see below) length:
    MPI_Type_vector(3, 1, 2, vec_struct_ty, &hvec_type);
    MPI_Type_commit(&hvec_type);

    struct VecT buffer[3] = {1., 2., 3., 4., 5., 6.};
    MPI_Send(&buffer[0], 1, hvec_type, 1, 0, MPI_COMM_WORLD);

    MPI_Type_free(&hvec_type);
  } else {
    double received[3];
    MPI_Recv(&received, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    fprintf(stderr, "[Test] Received: %.1f, %.1f, %.1f.\n", received[0], received[1], received[2]);
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
