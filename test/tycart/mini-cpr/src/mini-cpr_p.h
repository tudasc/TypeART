//
// Created by mority on 11/15/19.
// the private header file
//

#ifndef MINI_CPR_MINI_CPR_P_H
#define MINI_CPR_MINI_CPR_P_H

/**
 * struct representing a region in memory
 */
typedef struct mem_region {
    int id;
    void *start_addr;
    size_t count;
    size_t size;
} Mem_region;

/**
 * pointer to a mem_region
 */
typedef struct mem_region * Mem_region_ptr;

/*
 * struct for keeping track of memory regions to be protected
 */
typedef struct region_table {
    int size;
    int count;
    Mem_region_ptr *regions;
} Region_table;

/**
 * reads the config file
 * @param cfg_file the path to the config file
 * @return 0 if OK, else -1
 */
int mini_cpr_read_config(const char *cfg_file);

/**
 * removes leading and trailing whitespace from str
 * @param str a pointer to the '\0' terminated string
 * @return char* to the first non-white space character in str
 */
char* mini_cpr_remove_whitespace(char * str);

/**
 * @return the directory where the checkpoints are stored
 */
char* mini_cpr_checkpoint_dir();

/**
 * initializes the region table
 * @return 0 if OK, else -1
 */
int mini_cpr_rt_init();

/**
 * computes the auxiliary hash of the given id
 * @param id the id to be hashed
 * @return the auxiliary hash of the id
 */
int mini_cpr_rt_aux_hash(int id);

/**
 * computes the hash of the given id
 * @param id the id to be hashed
 * @param i the offset from the initial hash
 * @return the hash of id depending on the offset i
 */
int mini_cpr_rt_hash(int id, int i);

/**
 * searches the region table for an entry with given id
 * @param id the id the table will be searched for
 * @return the index of the entry for id in the region table, if no entry exists -1 is returned
 */
int mini_cpr_rt_search(int id);

/**
 * inserts an entry for a memory region into the region table
 * @param id integer label of the memory region, set by application
 * @param ptr pointer to the start address of the memory region
 * @param count the number of elements to be registered
 * @param size the size of an individual element
 * @return the index of the entry in the region table if successful, else -1
 */
int mini_cpr_rt_insert(int id, void * ptr, size_t count, size_t size);

/**
 * computes the load factor of the region table
 * @return the load factor, a value between 0 and 1
 */
float mini_cpr_rt_loadfactor();

/**
 * resizes the rt_table according to the resize factor
 * @return 0 if OK, else -1
 */
int mini_cpr_rt_resize();

/**
 * returns a pointer to the Mem_region in the region table specified by id
 * @param id integer label of the memory region, set by application
 * @return a pointer to the Mem_region, or NULL if no Mem_region with the given id exists
 */
Mem_region_ptr mini_cpr_rt_getMemregion(int id);

/**
 * creates a Mem_region struct on the heap an returns a pointer to it
 * @param id integer label of the memory region, set by application
 * @param ptr pointer to the start address of the memory region
 * @param count the number of elements to be registered
 * @param size the size of an individual element
 * @return a pointer to the created struct or NULL in case of failure
 */
Mem_region_ptr mini_cpr_makeMemregion(int id, void * ptr, size_t count, size_t size);

/**
 * updates the entry in the region table with the specified values
 * @param mrp pointer to the memory region to be updated
 * @param id integer label of the memory region, set by application
 * @param ptr pointer to the start address of the memory region
 * @param count the number of elements to be registered
 * @param size the size of an individual element
 * @return 0 if OK, else -1
 */
int mini_cpr_rt_update(Mem_region_ptr mrp, int id, void * ptr, size_t count, size_t size);

/**
 * opens a checkpoint file for i/o
 * @param name the name of the checkpoint, set by application
 * @param version version number of the checkpoint, set by application
 * @param mode mode passed to fopen(...), e.g. "rb", "wb" etc.
 * @return a stream to the named file, or NULL if the attempt fails
 */
FILE *mini_cpr_open_cp_file(const char *name, int version, const char *mode);

/**
 * serializes and writes a mem_region to a stream
 * @param cp_file stream of the checkpoint file where the mem_region will be written
 * @param mrp a pointer to the mem_region to be serialized and written to file
 * @return 0 if OK, else -1
 */
int mini_cpr_write_mem_region(FILE * cp_file, Mem_region_ptr mrp);

/**
 * reads a mem_region from a stream, data read is saved in the mem_region struct referenced by the given pointer
 * expects the current file position to be at the start, i.e. the id field, of a mem_region
 * @param cp_file the stream of the checkpoint file to read from
 * @param mrp pointer to the mem_region struct to save the data read
 * @return 0 if OK, else -1
 */
int mini_cpr_read_mem_region(FILE * cp_file, Mem_region_ptr mrp);

#endif //MINI_CPR_MINI_CPR_P_H
