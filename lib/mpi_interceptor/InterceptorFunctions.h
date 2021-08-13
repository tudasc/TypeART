#pragma once

#include <mpi.h>

#ifdef  __cplusplus
extern "C" {
#endif

void typeart_check_send(const char* name, const void* called_from, const void* sendbuf, int count, MPI_Datatype dtype);

void typeart_check_recv(const char* name, const void* called_from, void* recvbuf, int count, MPI_Datatype dtype);

void typeart_check_send_and_recv(const char* name, const void* called_from, const void* sendbuf, int sendcount,
                            MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype);

void typeart_unsupported_mpi_call(const char* name, const void* called_from);

void typeart_exit();

#ifdef  __cplusplus
}
#endif
