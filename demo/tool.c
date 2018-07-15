#include <mpi.h>
#include <stdio.h>

#include <RuntimeInterface.h>

int isCompatible(MPI_Datatype mpi_type, typeart_builtin_type recorded_type) {
  // This comparison is not exhaustive and is only used for this simple demo
  switch (recorded_type) {
    case TA_INT8:
      return mpi_type == MPI_CHAR || mpi_type == MPI_UNSIGNED_CHAR;
    case TA_INT16:
      return mpi_type == MPI_SHORT || mpi_type == MPI_UNSIGNED_SHORT;
    case TA_INT32:
      return mpi_type == MPI_INT || mpi_type == MPI_UNSIGNED;
    case TA_INT64:
      return mpi_type == MPI_LONG || mpi_type == MPI_UNSIGNED_LONG;
    case TA_FLOAT:
      return mpi_type == MPI_FLOAT;
    case TA_DOUBLE:
      return mpi_type == MPI_DOUBLE;
    default:
      break;
  }
  return 0;
}

void analyseBuffer(const void* buf, int count, MPI_Datatype type) {
  int num_integers, num_addresses, num_datatypes, combiner;
  PMPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);
  //  printf("MPI_Type_get_envelope(t,%i,%i,%i,%i)\n", num_integers, num_addresses, num_datatypes, combiner);
  int array_of_integers[num_integers];
  MPI_Aint array_of_addresses[num_addresses];
  MPI_Datatype array_of_datatypes[num_datatypes];
  if (combiner == MPI_COMBINER_NAMED) {
    int size;
    MPI_Type_size(type, &size);

    char type_name[MPI_MAX_OBJECT_NAME];
    int name_len;
    MPI_Type_get_name(type, type_name, &name_len);

    printf("Basetype(%s, addr=%p, size=%i , count=%i)\n", type_name, buf, size, count);

    typeart_type_info type_info;
    size_t count_check;
    typeart_status status = typeart_get_type(buf, &type_info, &count_check);

    if (status == TA_OK) {
      // If the address corresponds to a struct, fetch the type of the first member
      while (type_info.kind == STRUCT) {
        typeart_struct_layout struct_layout;
        typeart_resolve_type(type_info.id, &struct_layout);
        type_info = struct_layout.member_types[0];
      }

      // fprintf(stderr, "Type id=%d, name=%s\n", type_info.id, typeart_get_type_name(type_info.id));

      if (isCompatible(type, type_info.id)) {
        // printf("Types are compatible\n");
      } else {
        const char* recorded_name = typeart_get_type_name(type_info.id);
        if (type_info.kind == POINTER) {
          recorded_name = "Pointer";
        }

        fprintf(stdout, "Error: Incompatible buffer of type %d (%s) - expected %s instead\n", type_info.id,
                recorded_name, type_name);
      }

    } else {
      fprintf(stdout, "Error: ");
      if (status == TA_BAD_ALIGNMENT) {
        fprintf(stdout, "Buffer address does not align with the underlying type at %p\n", buf);
      } else if (status == TA_UNKNOWN_ADDRESS) {
        fprintf(stdout, "No buffer allocated at address %p\n", buf);
      }
    }

    return;
  }

  MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, array_of_integers, array_of_addresses,
                        array_of_datatypes);
  if (combiner == MPI_COMBINER_RESIZED) {  // MPI_TYPE_CREATE_RESIZED
    int i;
    MPI_Aint offset;
    for (i = 0, offset = array_of_addresses[0]; i < count; i++, offset += array_of_addresses[1])
      analyseBuffer((void*)((MPI_Aint)buf + offset), 1, array_of_datatypes[0]);
    return;
  }

  if (combiner == MPI_COMBINER_STRUCT) {  // MPI_TYPE_CREATE_STRUCT
    int i, j;
    MPI_Aint offset, lb, extent;
    MPI_Type_get_extent(type, &lb, &extent);

    fprintf(stdout, "Analyzing %d structs:\n", count);
    for (i = 0, offset = 0; i < count; i++, offset += extent) {
      for (j = 0; j < array_of_integers[0]; j++)
        analyseBuffer((void*)((MPI_Aint)buf + offset + array_of_addresses[j]), array_of_integers[j + 1],
                      array_of_datatypes[j]);
      fprintf(stdout, "\n");
    }

    return;
  }
}

int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf,
                 int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status* status) {
  printf("Analyze Send\n");
  analyseBuffer(sendbuf, sendcount, sendtype);
  // analyseBuffer(0, sendcount, sendtype);
  printf("Analyze Recv\n");
  analyseBuffer(recvbuf, recvcount, recvtype);
  // analyseBuffer(0, recvcount, recvtype);
  return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm,
                       status);
}

int MPI_Bcast(void* buf, int sendcount, MPI_Datatype sendtype, int root, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if( rank == root ) {
    printf("Analyze Send\n");
    analyseBuffer(buf, sendcount, sendtype);
//  analyseBuffer(0, sendcount, sendtype);
  } else {
    printf("Analyze Recv\n");
    analyseBuffer(buf, sendcount, sendtype);
//  analyseBuffer(0, recvcount, recvtype);
  }
  return PMPI_Bcast(buf, sendcount, sendtype, root, comm);
}


