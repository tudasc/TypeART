#include <RuntimeInterface.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct test{
  int a;
  double b;
  unsigned int c;
  void* d;
  int e;
  struct test* f;
};


int main(int argc, char** argv){
  MPI_Init(&argc, &argv);

#ifdef NOSTACK
  struct test * mystruct=malloc(sizeof(struct test)*5);
#else
  struct test mystruct[5];
#endif

  typeart_type_info type;
  typeart_builtin_type btype;
  size_t count;
  
#ifdef NOSTACK
  int* buffer = malloc(sizeof(int)*50);
#else
  int buffer[50];
#endif

  typeart_status status = typeart_get_type(mystruct, &type, &count);
  
  if (status==TA_OK) printf("type (kind=%i,id=%i), count=%lu\n", type.kind, type.id, count);
  else printf("error: %i\n", status);
  
  status = typeart_get_builtin_type(&(mystruct[2].e), &btype);
  
  if (status==TA_OK) printf("type (kind=%i)\n", btype);
  else printf("error: %i\n", status);
  
  status = typeart_get_type(&(mystruct[2].e), &type, &count);
  
  if (status==TA_OK) printf("type (kind=%i,id=%i), count=%lu\n", type.kind, type.id, count);
  else printf("error: %i\n", status);
  
  status = typeart_get_builtin_type(&(mystruct[2].c), &btype);
  
  if (status==TA_OK) printf("type (kind=%i)\n", btype);
  else printf("error: %i\n", status);
  
  status = typeart_get_type(&(mystruct[2].c), &type, &count);
  
  if (status==TA_OK) printf("type (kind=%i,id=%i), count=%lu\n", type.kind, type.id, count);
  else printf("error: %i\n", status);
  
  const void* base_address;
  size_t offset;
  status = typeart_get_containing_type(&(mystruct[2].c), &type, &count, &base_address, &offset);
  
  if (status==TA_OK) printf("containing_type (kind=%i,id=%i), count=%lu, %p, %lu\n", type.kind, type.id, count, base_address, offset);
  else printf("error: %i\n", status);
  
  printf("buffer\n");
  
  status = typeart_get_containing_type(&(buffer[20]), &type, &count, &base_address, &offset);
  
  if (status==TA_OK) printf("containing_type (kind=%i,id=%i), count=%lu, %p, %lu\n", type.kind, type.id, count, base_address, offset);
  else printf("error: %i\n", status);
  
  MPI_Sendrecv(mystruct, 1, MPI_INT, 0, 0, buffer+20, 1, MPI_INT,
                 0,0, MPI_COMM_SELF, MPI_STATUS_IGNORE);

  MPI_Finalize();
}