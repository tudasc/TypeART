
#include "tycart.h"
#include "Runtime.h"

#include "mpi.h"

#include "veloc.h"
#include "typecheck_macro.h"


int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  
  VELOC_Init(MPI_COMM_WORLD, "../gh-cdrischler-eos-mbpt/veloc.cfg");

  int a;
  a = 0;
  int b = 1;
  int *c = (int*) malloc(5*sizeof(int));

  __typeart_alloc_stack(&a, 2, 1);
  __typeart_alloc_stack(&b, 2, 1);
  __typeart_alloc(c, 2, 5);

  // type id 2 = int32
  __tycart_assert(0, &a, 1, sizeof(int), 2);
  __tycart_assert(1, &b, 1, sizeof(int), 2);
  __tycart_assert(2, c, 5, sizeof(int), 2);

  //free(c);
  //__typeart_free(c);
  //c = (int *) realloc(c, 2*sizeof(int));
  //__typeart_alloc(c, 2, 2);
 
  __tycart_cp_assert();

  __typeart_leave_scope(1);

  VELOC_Finalize(0);
  MPI_Finalize();
  
  return 0;
}
