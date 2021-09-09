// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | FileCheck %s
// RUN: cat "%s.log/1/rank.1/stderr" | FileCheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  double send_buffer[2] = {1.0};
  int send_count[2]     = {0};
  int send_displys[2]   = {0};

  double recv_buffer[2] = {0.0};
  int recv_count[2]     = {0};
  int recv_displys[2]   = {0};

  MPI_Datatype mpi_type[2] = {MPI_DOUBLE, MPI_DOUBLE};

  MPI_Request mpi_req[2];

  // clang-format off
  MPI_Alltoallw(send_buffer, send_count, send_displys, mpi_type,
                recv_buffer, recv_count, recv_displys, mpi_type,
                MPI_COMM_WORLD);

  MPI_Ialltoallw(send_buffer, send_count, send_displys, mpi_type,
                 recv_buffer, recv_count, recv_displys, mpi_type,
                 MPI_COMM_WORLD, &mpi_req[0]);
  // clang-format on

  MPI_Barrier(MPI_COMM_WORLD);

  // CHECK: Unsupported: 2

  MPI_Finalize();
  return 0;
}