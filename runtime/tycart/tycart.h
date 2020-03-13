#ifndef TYCART_RT_H
#define TYCART_RT_H

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif // __cplusplus

  /*
   * In TyCart this function first checks the assert, then registers the memory region
   * into the TyCart runtime table, i.e., we can run the asserts before a checkpoint
   * again.
   * Then calls the target-CPR library specific register function.
   */
  void __tycart_assert(int id, void *addr, size_t count, size_t typeSize, int typeId);

  /*
   * Iterates the stored CP map, to re-assert all stored assumptions before calling the final
   * checkpointing mechanism.
   */
  void __tycart_cp_assert();

  /*
   * Given the type ID, it registers the FTI user-defined type to our Runtimne
   */
  void __tycart_register_FTI_t(int typeId);


#ifdef __cplusplus
}
#endif // __cplusplus

#endif
