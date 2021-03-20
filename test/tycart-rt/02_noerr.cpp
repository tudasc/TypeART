
#include "tycart.h"
#include "Runtime.h"
#include "CallbackInterface.h"

#include "mpi.h"


#ifdef WITH_VELOC
#include "veloc.h"
#endif
#ifdef WITH_FTI
#include "fti.h"
#endif
#include "typecheck_macro.h"


int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

#ifdef WITH_VELOC  
  VELOC_Init(MPI_COMM_WORLD, "./veloc.cfg");
#endif

#ifdef WITH_FTI
  FTI_Init("./config.fti", MPI_COMM_WORLD);
#endif

  int a;
  a = 0;

  __typeart_alloc_stack(&a, 2, 1);

  // type id 2 = int32
  __tycart_assert(0, &a, 1, sizeof(int), 2);

  __tycart_cp_assert();

  __typeart_leave_scope(1);

#ifdef WITH_VELOC
  VELOC_Finalize(0);
#endif

#ifdef WITH_FTI
  FTI_Finalize();
#endif

  MPI_Finalize();
  
  return 0;
}
