#include "SerialTemplateGoL.h"

//*************************
// VELOC Checkpoint - Restart
//*************************
#include <../../../include/VELOC/include/veloc.h>

#define CHKP_INTERVAL 20

//*************************
// Type-Assert Macro
//*************************
#include "../../../include/TypeART/runtime/typecheck_macro.h"

//*************************
// Testing Macros
//*************************
#include "../../../test/gol_testing.h"

struct InitFunc {
public:
  double operator()(int i, int j) {
    return (double(i) * i - i * j / (i + j + 1.0));
  }
};

int main(int argc, char **argv) {

  // initialize GameOfLife board
  GameOfLife<ACTUAL_TYPE, GoLStencil<ACTUAL_TYPE>> gol(ACTUAL_X, ACTUAL_Y);
  MyInit f;
  gol.init(f);

  // initialize CPR lib
  MPI_Init(&argc, &argv);
  VELOC_Init(MPI_COMM_WORLD, "veloc.config");

  // register critical variables for checkpointing
  // TY_protect(0, &(gol.dimX), 1, int); // causes a false positive
  TY_protect(1, &(gol.dimY), 1, int);
  TY_protect(2, gol.gridA.data(), ACTUAL_SIZE, ACTUAL_TYPE);
  TY_protect(3, gol.gridB.data(), EXPECTED_SIZE, EXPECTED_TYPE);

  // check for restart
  int r_check = VELOC_Restart_test("instrumented_veloc_gol", 0);
  int i = 0;
  if (r_check != VELOC_FAILURE) {
    // restart
    i = r_check;
    printf("------ Restarting ------ Version %d ------\n", r_check);
    VELOC_Restart("instrumented_veloc_gol", r_check);
    printf("Grid loaded from checkpoint:\n");
    gol.print(std::cout);
  }

  //	gol.print(std::cout);
  printf("------ Starting Main Loop ------ Iteration %d ------\n", i);
  for (; i < 100; ++i) {
    gol.tick();
    if ((i + 1) % CHKP_INTERVAL == 0) {
      printf("------ Checkpoint ------ Iteration %d ------\n", i);
      VELOC_Checkpoint("instrumented_veloc_gol", (i + 1));
    }
  }

  printf("------ Final Grid ------ Iteration %d ------\n", i);
  gol.print(std::cout);

  // finalize CPR lib
  VELOC_Finalize(0); // 0 means keep checkpoints

  MPI_Finalize();
  return 0;
}
