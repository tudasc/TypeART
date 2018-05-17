#include <stdio.h>
#include <mpi.h>

#include <RuntimeInterface.h>

void analyseBuffer(const void *buf, int count, MPI_Datatype type)
{
  int num_integers, num_addresses, num_datatypes, combiner;
  PMPI_Type_get_envelope( type, &num_integers, &num_addresses, &num_datatypes, &combiner);
//  printf("MPI_Type_get_envelope(t,%i,%i,%i,%i)\n", num_integers, num_addresses, num_datatypes, combiner);
  int array_of_integers[num_integers];
  MPI_Aint array_of_addresses[num_addresses];
  MPI_Datatype array_of_datatypes[num_datatypes];
  if (combiner==MPI_COMBINER_NAMED)
  {
    int size;
    MPI_Type_size(type, &size);
    printf("Basetype(t, addr=%p, size=%i , count=%i)\n", buf, size, count);
    //TODO: check for matching c-type

    must_type_info type_info;
    int count_check;
    lookup_result status = must_support_get_type(buf, &type_info, &count_check);
    if (status == SUCCESS) {
      
      printf("Lookup was successful!\n");
      printf("Type name: %s\n", must_support_get_type_name(type_info.id));
    } else {
      printf("Lookup failed: ");
      if (status == BAD_ALIGNMENT) {
        printf("Bad alignment\n");
      } else if (status == UNKNOWN_ADDRESS) {
        printf("Unknown address\n");
      }
    }

    return;
  }
  
  MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, 
                        array_of_integers, array_of_addresses, array_of_datatypes);
  if (combiner==MPI_COMBINER_RESIZED)
  {// MPI_TYPE_CREATE_RESIZED
    int i; MPI_Aint offset;
    for (i=0, offset=array_of_addresses[0]; i<count; i++, offset+=array_of_addresses[1] )
      analyseBuffer((void*)((MPI_Aint)buf+offset), 1, array_of_datatypes[0]);
    return;
  }
    
  if (combiner==MPI_COMBINER_STRUCT)
  {// MPI_TYPE_CREATE_STRUCT
    int i, j; MPI_Aint offset, lb, extent;
    MPI_Type_get_extent(type, &lb, &extent);
    for (i=0, offset=0; i<count; i++, offset+=extent )
      for (j=0; j<array_of_integers[0]; j++)
        analyseBuffer((void*)((MPI_Aint)buf+offset+array_of_addresses[j]), array_of_integers[j+1], array_of_datatypes[j]);
    return;
  }
    
}


int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                int dest, int sendtag,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int source, int recvtag,
                MPI_Comm comm, MPI_Status *status)
{
  printf("Analyze Send\n");
  analyseBuffer(sendbuf, sendcount, sendtype);
  analyseBuffer(0, sendcount, sendtype);
  printf("Analyze Recv\n");
  analyseBuffer(recvbuf, recvcount, recvtype);
  analyseBuffer(0, recvcount, recvtype);
  PMPI_Sendrecv (sendbuf, sendcount, sendtype, dest, sendtag,
                recvbuf, recvcount, recvtype, source, recvtag,
                 comm, status);
}

