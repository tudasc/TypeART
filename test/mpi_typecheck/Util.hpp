#ifndef TYPEART_TEST_MPI_INTERCEPTOR_UTIL_H
#define TYPEART_TEST_MPI_INTERCEPTOR_UTIL_H

#include <mpi.h>

template <size_t N>
struct padded_array {
  double offset;  // needed, cause otherwise the TypeART type of arr will be struct
  double arr[N];
  double padding;
};

inline void run_test(void* data, int count, MPI_Datatype type) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    MPI_Send(data, count, type, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(data, count, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

template <size_t N>
inline void run_test(padded_array<N>& data, int count, MPI_Datatype type) {
  run_test(data.arr, count, type);
}

#endif  // TYPEART_TEST_MPI_INTERCEPTOR_UTIL_H
