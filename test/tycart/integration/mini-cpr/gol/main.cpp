#include "SerialTemplateGoL.h"

//*************************
// mini-cpr Checkpoint - Restart
//*************************
#include "mini-cpr.h"

#define CHKP_INTERVAL 20

//*************************
// Type-Assert Macro
//*************************
#include "../../../../../lib/runtime/tycart/typecheck_macro.h"

//*************************
// Testing Macros
//*************************
#include "../../gol_testing.h"

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
  mini_cpr_init("mini-cpr.config");

  // register critical variables for checkpointing
  //TY_protect(0, &(gol.dimX), 1, int); // causes a false positive
  TY_protect(1, &(gol.dimY), 1, int);
  TY_protect(2, gol.gridA.data(), ACTUAL_SIZE, ACTUAL_TYPE);
  TY_protect(3, gol.gridB.data(), EXPECTED_SIZE, EXPECTED_TYPE);

  // check for restart
  int r_check = mini_cpr_restart_check("instrumented_mini-cpr_gol", 100);
  int i = 0;
  if (r_check != -1) {
    // restart
    i = r_check;
    printf("------ Restarting ------ Version %d ------\n", r_check);
    mini_cpr_restart("instrumented_mini-cpr_gol", r_check);
    printf("Grid loaded from checkpoint:\n");
    gol.print(std::cout);
  }

  //	gol.print(std::cout);
  printf("------ Starting Main Loop ------ Iteration %d ------\n", i);
  for (; i < 100; ++i) {
    gol.tick();
    if ((i + 1) % CHKP_INTERVAL == 0) {
      printf("------ Checkpoint ------ Iteration %d ------\n", i);
      mini_cpr_checkpoint("instrumented_mini-cpr_gol", (i + 1));
    }
  }

  printf("------ Final Grid ------ Iteration %d ------\n", i);
  gol.print(std::cout);

  // finalize CPR lib
  mini_cpr_fin();

  return 0;
}
