#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MSG_TAG 666
#define COUNT   5

// Comment out for heap allocation
//#define USE_STACK

struct particle {
  int id;
  double vel[3];
  double pos[3];
  struct particle* next;
};

void resetParticles(struct particle* p, int count) {
  int i, j;
  for (i = 0; i < count; i++) {
    for (j = 0; j < 3; j++) {
      p[i].vel[j] = 0;
      p[i].pos[j] = 0;
    }
    p[i].id = 0;
  }
}

void printParticles(struct particle* p, int count, int rank) {
  int i;
  for (i = 0; i < count; i++) {
    printf("%i: (%lf,\t%lf,\t%lf),\t(%lf,\t%lf,\t%lf), %i\n", rank, p[i].vel[0], p[i].vel[1], p[i].vel[2], p[i].pos[0],
           p[i].pos[1], p[i].pos[2], p[i].id);
  }
}

int main(int argc, char** argv) {
  int size    = -1;
  int my_rank = -1, i, j;

  int array_of_blocklengths[COUNT] = {1, 1, 3, 3, 1};

#ifdef USE_STACK
  struct particle localParticles[COUNT], remoteParticles[COUNT];
#else
  struct particle* localParticles  = malloc(sizeof(struct particle) * COUNT);
  struct particle* remoteParticles = malloc(sizeof(struct particle) * COUNT);
#endif

  MPI_Aint array_of_displacements[COUNT];
  MPI_Aint array_of_some_displacements[COUNT];
  MPI_Aint first_var_address;
  MPI_Aint second_var_address;

  MPI_Datatype array_of_types[COUNT - 2] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
  MPI_Datatype parttype, fulltype, veltype, postype;

  MPI_Status status;
  MPI_Request request;

  /* Get process and neighbor info. */
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf(" I am rank %d of %d PEs\n", my_rank, size);

  for (i = 0; i < COUNT; i++) {
    for (j = 0; j < 3; j++) {
      localParticles[i].vel[j] = my_rank * 10000. + i * 100 + j + 1;
      localParticles[i].pos[j] = 1000000. + my_rank * 10000. + i * 100 + j + 1;
    }
    localParticles[i].id = i;
  }
  printParticles(localParticles, COUNT, my_rank);
  /* Construct MPI datatypes for sending and receiving particle information. */

  array_of_displacements[0] = 0;
  array_of_displacements[1] = array_of_displacements[0];
  array_of_displacements[2] = array_of_displacements[1] + sizeof(int);
  array_of_displacements[3] = array_of_displacements[2] + sizeof(double) * 3;
  array_of_displacements[4] = array_of_displacements[3] + sizeof(double) * 3 + sizeof(void*);

  MPI_Type_create_struct(COUNT - 2, array_of_blocklengths + 1, array_of_displacements + 1, array_of_types, &parttype);
  MPI_Type_create_resized(parttype, array_of_displacements[0], array_of_displacements[COUNT - 1], &fulltype);
  MPI_Type_commit(&fulltype);
  MPI_Type_free(&parttype);

  MPI_Type_create_struct(1, array_of_blocklengths + 2, array_of_displacements + 2, array_of_types + 1, &parttype);
  MPI_Type_create_resized(parttype, array_of_displacements[0], array_of_displacements[COUNT - 1], &veltype);
  MPI_Type_commit(&veltype);
  MPI_Type_free(&parttype);

  MPI_Type_create_struct(1, array_of_blocklengths + 3, array_of_displacements + 3, array_of_types + 2, &parttype);
  MPI_Type_create_resized(parttype, array_of_displacements[0], array_of_displacements[COUNT - 1], &postype);
  MPI_Type_commit(&postype);
  MPI_Type_free(&parttype);

  MPI_Aint lb, extent;
  MPI_Type_get_extent(fulltype, &lb, &extent);
  // printf("fulltype lb: %li, extent: %li\n", lb, extent);
  MPI_Type_get_extent(veltype, &lb, &extent);
  // printf("veltype lb: %li, extent: %li\n", lb, extent);
  MPI_Type_get_extent(postype, &lb, &extent);
  // printf("postype lb: %li, extent: %li\n", lb, extent);

  printf("Sendrecv1:\n");
  MPI_Sendrecv(localParticles, COUNT, fulltype, size - my_rank - 1, MSG_TAG, remoteParticles, COUNT, fulltype,
               size - my_rank - 1, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printParticles(remoteParticles, COUNT, my_rank);
  resetParticles(remoteParticles, COUNT);

  printf("Sendrecv2:\n");
  MPI_Sendrecv(localParticles, COUNT, veltype, size - my_rank - 1, MSG_TAG, remoteParticles, COUNT, veltype,
               size - my_rank - 1, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printParticles(remoteParticles, COUNT, my_rank);
  resetParticles(remoteParticles, COUNT);

  printf("Sendrecv3:\n");
  MPI_Sendrecv(localParticles, COUNT, postype, size - my_rank - 1, MSG_TAG, remoteParticles, COUNT, postype,
               size - my_rank - 1, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printParticles(remoteParticles, COUNT, my_rank);
  resetParticles(remoteParticles, COUNT);

  MPI_Type_free(&fulltype);
  MPI_Type_free(&veltype);
  MPI_Type_free(&postype);

#ifndef USE_STACK
  free(localParticles);
  free(remoteParticles);
#endif

  MPI_Finalize();

  return 0;
}
