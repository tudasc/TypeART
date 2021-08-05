#include <RuntimeInterface.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

struct test {
  int a;
  double b;
  unsigned int c;
  void* d;
  int e;
  struct test* f;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

#ifdef NOSTACK
  struct test* mystruct = malloc(sizeof(struct test) * 5);
#else
  struct test mystruct[5];
#endif

  int type     = 0;
  size_t count = 0;

#ifdef NOSTACK
  int* buffer = malloc(sizeof(int) * 50);
#else
  int buffer[50];
#endif

  typeart_status status = typeart_get_type(mystruct, &type, &count);

  if (status == TYPEART_OK)
    printf("type (id=%i), count=%lu\n", type, count);
  else
    printf("[Demo] Toy Error: %i\n", status);

  status = typeart_get_type(&(mystruct[2].e), &type, &count);

  if (status == TYPEART_OK)
    printf("(sub) type (id=%i), count=%lu\n", type, count);
  else
    printf("[Demo] Toy Error: %i\n", status);

  status = typeart_get_type(&(mystruct[2].e), &type, &count);

  if (status == TYPEART_OK)
    printf("type (id=%i), count=%lu\n", type, count);
  else
    printf("[Demo] Toy Error: %i\n", status);

  status = typeart_get_type(&(mystruct[2].c), &type, &count);

  if (status == TYPEART_OK)
    printf("(sub) type (id=%i), count=%lu\n", type, count);
  else
    printf("[Demo] Toy Error: %i\n", status);

  status = typeart_get_type(&(mystruct[2].c), &type, &count);

  if (status == TYPEART_OK)
    printf("type (id=%i), count=%lu\n", type, count);
  else
    printf("[Demo] Toy Error: %i\n", status);

  const void* base_address;
  size_t offset;
  status = typeart_get_containing_type(&(mystruct[2].c), &type, &count, &base_address, &offset);

  if (status == TYPEART_OK)
    printf("containing_type (id=%i), count=%lu, %p, %lu\n", type, count, base_address, offset);
  else
    printf("[Demo] Toy Error: %i\n", status);

  printf("buffer\n");

  status = typeart_get_containing_type(&(buffer[20]), &type, &count, &base_address, &offset);

  if (status == TYPEART_OK)
    printf("containing_type (id=%i), count=%lu, %p, %lu\n", type, count, base_address, offset);
  else
    printf("[Demo] Toy Error: %i\n", status);

  MPI_Sendrecv(mystruct, 1, MPI_INT, 0, 0, buffer + 20, 1, MPI_INT, 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE);

  MPI_Finalize();

#ifdef NOSTACK
  free(mystruct);
#else
#endif

#ifdef NOSTACK
  free(buffer);
#else
#endif
}
