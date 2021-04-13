//
// Created by mority on 11/15/19.
// inspired by VeloC
//

#ifndef MINI_CPR_MINI_CPR_H
#define MINI_CPR_MINI_CPR_H

#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * initializes the checkpoint/restart system
     * @param cfg_file path to config file, containing the path to the directory where checkpoints will be stored
     * @return 0 if OK, else -1
     */
    int mini_cpr_init(const char *cfg_file);

    /**
     * clean-up routine
     */
    void mini_cpr_fin();

    /**
     * registers a memory region with the checkpoint/restart system
     * @param id integer label of the memory region, set by application
     * @param ptr pointer to the start address of the memory region
     * @param count the number of elements to be registered
     * @param size the size of an individual element
     * @return 0 if OK, else -1
     */
    int mini_cpr_register(int id, void * ptr, size_t count, size_t size);

    /**
     * unregisters a memory region from the checkpoint/restart system
     * @param id integer label of the memory region, set by application
     * @return 0 if OK, else -1
     */
    int mini_cpr_unregister(int id);

    /**
     * creates a checkpoint of the registered memory regions
     * @param name the name of the checkpoint, set by application
     * @param version version number of the checkpoint >= 0, set by application
     * @return 0 if OK, else -1
     */
    int mini_cpr_checkpoint(const char *name, int version);

    /**
     * checks if a checkpoint exists
     * @param name the name of the checkpoint, set by application
     * @param version the maximum version number to look for (>= 0)
     * @return the version number of the checkpoint found, -1 if no checkpoint could be found
     */
    int mini_cpr_restart_check(const char *name, int version);

    /**
     * loads the registered memory regions from the checkpoint
     * @param name the name of the checkpoint, set by application
     * @param version version number of the checkpoint (>= 0)
     * @return 0 if OK, else -1
     */
    int mini_cpr_restart(const char *name, int version);


#ifdef __cplusplus
}
#endif

#endif //MINI_CPR_MINI_CPR_H

