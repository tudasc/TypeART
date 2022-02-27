// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 %s.exe" 2>&1 | %filecheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include <mpi.h>
#include <stdio.h>

// CHECK-DAG: MPI_Recv: successfully checked recv-buffer
// CHECK-DAG: MPI_Send: successfully checked send-buffer

void run(int rank) {
#define BUFF_SIZE 18
  int buffer[BUFF_SIZE];

  MPI_Datatype contig_ty;
  MPI_Type_contiguous(3, MPI_INT, &contig_ty);
  MPI_Type_commit(&contig_ty);

  MPI_Datatype hvec_ty;
  MPI_Type_create_hvector(3, 1, 6 * sizeof(int), contig_ty, &hvec_ty);
  MPI_Type_commit(&hvec_ty);

  if (rank == 0) {
    for (int i = 0; i < BUFF_SIZE; ++i) {
      buffer[i] = -1;
    }

    MPI_Send(buffer, 1, hvec_ty, 1, 0, MPI_COMM_WORLD);
  } else {
    for (int i = 0; i < BUFF_SIZE; ++i) {
      buffer[i] = 0;
    }

    MPI_Recv(buffer, 1, hvec_ty, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < BUFF_SIZE; ++i) {
      printf("%i ", buffer[i]);
    }
    printf("\n");
  }

  MPI_Type_free(&hvec_ty);
  MPI_Type_free(&contig_ty);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  run(rank);

  MPI_Finalize();
  return 0;
}