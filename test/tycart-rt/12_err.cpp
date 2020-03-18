
#include "Runtime.h"

#include "mpi.h"


#ifdef WITH_VELOC
#include "veloc.h"
#endif
#ifdef WITH_FTI
#include "fti.h"
#endif

//#define TYCART_TEST_ true
#include "tycart.h"


int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

#ifdef WITH_VELOC  
  VELOC_Init(MPI_COMM_WORLD, "./veloc.cfg");
#endif

#ifdef WITH_FTI
  FTI_Init("./config.fti", MPI_COMM_WORLD);
#endif

  float a;
  a = 0.0f;

  __typeart_alloc_stack(&a, 5, 1);

  // type id 2 = int32
  // In test with the Macro, we can only assert int
  TY_protect(0, &a, 1, int)

  TY_checkpoint("test", 0, 1, 1)

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
