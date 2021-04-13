//
// Created by mority on 11/15/19.
//


#include "values.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#include "assert.h"

#include "mini-cpr.h"
#include "mini-cpr_p.h"

#define PATH_LENGTH 4096
#define INIT_TABLE_SIZE 128
#define MAX_LOAD 0.9
#define RESIZE_FACTOR 2
#define DELETED INT_MIN

/*
 * maintains a table of memory regions
 */
Region_table rt;

/*
 * the directory where checkpoints are stored
 */
char * checkpoint_dir = NULL;


int mini_cpr_read_config(const char *cfg_file) {
    FILE *fp;
    fp = fopen(cfg_file, "r");
    if(fp == NULL) {
        return -1;
    }

    char buff[PATH_LENGTH];
    if(fgets(buff, PATH_LENGTH, fp) == NULL) {
        return -1;
    }

    checkpoint_dir = (char*)malloc(sizeof(char) * PATH_LENGTH);
    if(checkpoint_dir == NULL) {
        return -1;
    }

    strcpy(checkpoint_dir, mini_cpr_remove_whitespace(buff));
    if(checkpoint_dir == NULL) {
        return -1;
    }

    fclose(fp);
    return 0;
}

char* mini_cpr_remove_whitespace(char * str) {
  // pointer after leading whitespace
  char * firstNonWhite = NULL;
  // remove trailing whitespace
  int lastNonWhite = -1;
  for(int i = 0; str[i] != '\0'; i++) {
    if(str[i] != ' ' && str[i] != '\t' && str[i] != '\n') {
      if(firstNonWhite == NULL) {
        firstNonWhite = &(str[i]);
      }
      lastNonWhite = i;
    }
  }
  // set character after last non-whitespace character to '\0'
  str[lastNonWhite + 1] = '\0';
  if(!firstNonWhite) {
    printf("[mini-cpr] Error: Checkpoint path is empty\n");
  }
  return firstNonWhite;
}

char* mini_cpr_checkpoint_dir() {
    return checkpoint_dir;
}

int mini_cpr_rt_init() {
    rt.size = INIT_TABLE_SIZE;
    rt.count = 0;
    rt.regions = (Mem_region_ptr*) calloc(INIT_TABLE_SIZE, sizeof(Mem_region_ptr));
    if(rt.regions == NULL) {
        return -1;
    }
    return 0;
}

int mini_cpr_init(const char *cfg_file) {

    // read config file
    if(mini_cpr_read_config(cfg_file)) {
        printf("[mini-cpr] Error: Could not read configuration file %s\n", cfg_file);
        return -1;
    }

    // init region table
    if(mini_cpr_rt_init()) {
        printf("[mini-cpr] Error: Could not initialize table of memory regions\n");
        return -1;
    }

    printf("[mini-cpr] Info: Initialized. Table size is %d. Checkpoint directory is %s\n", rt.size,checkpoint_dir);
    return 0;
}

// TODO fixme, double free or corruption, not sure what's wrong, yet
void mini_cpr_fin() {
    free(checkpoint_dir);
    for(size_t i = 0; i < rt.size; ++i) {
        if(rt.regions[i] != NULL) {
            free(rt.regions[i]);
        }
    }
    free(rt.regions);
    printf("[mini-cpr] Info: Finalized\n");
}

int mini_cpr_rt_aux_hash(int id) {
    return id % rt.size;
}

int mini_cpr_rt_hash(int id, int i) {
    return (mini_cpr_rt_aux_hash(id) + i) % rt.size;
}

int mini_cpr_rt_search(int id) {
    int i = 0;
    while(1) {
        int hash = mini_cpr_rt_hash(id, i);
        if(rt.regions[hash] == NULL) {
            return -1;
        }
        if(rt.regions[hash]->id == id) {
            return hash;
        }
        ++i;
        if(i == rt.size) {
            return -1;
        }
    }
}

int mini_cpr_rt_insert(int id, void * ptr, size_t count, size_t size) {
    int i = 0;
    while(1) {
        int hash = mini_cpr_rt_hash(id, i);
        if(rt.regions[hash] == NULL) {
            rt.regions[hash] = mini_cpr_makeMemregion(id, ptr, count, size);
            ++rt.count;
            return hash;
        } else if(rt.regions[hash]->id == DELETED) {
            mini_cpr_rt_update(rt.regions[hash], id, ptr, count, size);
            ++rt.count;
            return hash;
        } else {
            ++i;
        }
        if (i == rt.size) {
            break;
        }
    }
    return -1;
}

float mini_cpr_rt_loadfactor() {
    return (float) rt.count / (float) rt.size;
}

Mem_region_ptr mini_cpr_makeMemregion(int id, void * ptr, size_t count, size_t size) {
    Mem_region_ptr mrp = (Mem_region_ptr) malloc(sizeof(Mem_region));
    if(mrp == NULL) {
        return NULL;
    }
    mrp->id = id;
    mrp->start_addr = ptr;
    mrp->count = count;
    mrp->size = size;
    return mrp;
}

int mini_cpr_rt_resize() {
    Mem_region_ptr *old_regions = rt.regions;
    int old_size = rt.size;
    rt.size = old_size * RESIZE_FACTOR;
    printf("[mini-cpr] Info: Resizing region table from %d by factor %d to %d\n", old_size, RESIZE_FACTOR, rt.size);
    rt.regions = (Mem_region_ptr*) calloc( (size_t)rt.size, sizeof(Mem_region_ptr));
    if(rt.regions == NULL) {
        return -1;
    }
    rt.count = 0;
    for(int i = 0; i < old_size; ++i) {
        if(old_regions[i] != NULL && old_regions[i]->id != DELETED) {
            mini_cpr_rt_insert(old_regions[i]->id, old_regions[i]->start_addr, old_regions[i]->count, old_regions[i]->size);
            free(old_regions[i]);
        }
    }
    free(old_regions);
    return 0;
}

Mem_region_ptr mini_cpr_rt_getMemregion(int id) {
    int ret = mini_cpr_rt_search(id);
    if(ret == -1) {
        return NULL;
    } else {
        return rt.regions[ret];
    }
}

int mini_cpr_rt_update(Mem_region_ptr mrp, int id, void * ptr, size_t count, size_t size) {
    if(mrp != NULL) {
        mrp->id = id;
        mrp->start_addr = ptr;
        mrp->count = count;
        mrp->size = size;
        return 0;
    } else {
        return -1;
    }
}

int mini_cpr_register(int id, void * ptr, size_t count, size_t size) {
    // check if resize of table necessary
    if(mini_cpr_rt_loadfactor() > MAX_LOAD) {
        if(mini_cpr_rt_resize()) {
            return -1;
        }
    }

    // entry already exists --> update, else insert
    int found_hash = mini_cpr_rt_search(id);
    if(found_hash != -1) {
        if(mini_cpr_rt_update(rt.regions[found_hash], id, ptr, count, size)) {
            return -1;
        }
    } else {
        if(mini_cpr_rt_insert(id, ptr, count, size) == -1) {
            return -1;
        }
    }
    return 0;
}

int mini_cpr_unregister(int id) {
    int found_hash = mini_cpr_rt_search(id);
    if(found_hash != -1) {
        rt.regions[found_hash]->id = DELETED;
        --rt.count;
        return 0;
    } else {
        return -1;
    }
}

FILE * mini_cpr_open_cp_file(const char *name, int version, const char *mode) {
    char version_str[21];
    snprintf(version_str, 21, "%d", version);
    //char * checkpoint_file_name = (char *) calloc((strlen(checkpoint_dir) + 1 + strlen(name) + strlen(version_str) + 1) , sizeof(char));
    char checkpoint_file_name[PATH_LENGTH] = "";
    strcat(checkpoint_file_name, checkpoint_dir);
    strcat(checkpoint_file_name, "/");
    strcat(checkpoint_file_name, name);
    strcat(checkpoint_file_name, version_str);
    if(mode[0] == 'w') {
      printf("[mini-cpr] Info: Attempting to write checkpoint file: %s\n",
             checkpoint_file_name);
    }
    return fopen(checkpoint_file_name, mode);
}

int mini_cpr_write_mem_region(FILE * cp_file, Mem_region_ptr mrp) {
    // write id
    assert(&mrp->id && "mrp->id valid");
    if(fwrite(&mrp->id, sizeof(int), 1, cp_file) != 1) {
        return -1;
    }
    // write count
    assert(&mrp->count && "mrp->count valid");
    if(fwrite(&mrp->count, sizeof(size_t), 1, cp_file) != 1) {
        return -1;
    }
    // write size
    assert(&mrp->size && "mrp->size valid");
    if(fwrite(&mrp->size, sizeof(size_t), 1, cp_file) != 1) {
        return -1;
    }
    // write actual content of memory region
    size_t ret = fwrite(mrp->start_addr, mrp->size, mrp->count, cp_file);
    if(ret != mrp->count) {
        printf("[mini-cpr] Error: When checkpointing memory region with id = %d, attempted to write %lu objects of size %lu bytes to checkpoint file, but could only write %lu objects\n", mrp->id, mrp->count, mrp->size, ret);
        return -1;
    }
    return 0;
}

int mini_cpr_checkpoint(const char *name, int version) {
    FILE * cp_file = mini_cpr_open_cp_file(name, version, "w");
    if(cp_file) {
        setbuf(cp_file, NULL);
        for (int i = 0; i < rt.size; ++i) {
            if (rt.regions[i] != NULL && rt.regions[i]->id != DELETED) {
                if (mini_cpr_write_mem_region(cp_file, rt.regions[i])) {
                    printf("[mini-cpr] Error: Could not checkpoint memory region %d\n", i);
                    return -1;
                }
            }
        }
        fclose(cp_file);
    } else {
        printf("[mini-cpr] Error: Could not open checkpoint file\n");
        return -1;
    }
    printf("[mini-cpr] Info: Completed checkpoint labeled %s with version = %d\n", name, version);
    return 0;
}

int mini_cpr_read_mem_region(FILE * cp_file, Mem_region_ptr mrp) {
    if(fread(&mrp->id, sizeof(int), 1, cp_file) != 1) {
        return -1;
    }
    if(fread(&mrp->count, sizeof(size_t), 1, cp_file) != 1) {
        return -1;
    }
    if(fread(&mrp->size, sizeof(size_t), 1, cp_file) != 1) {
        return -1;
    }
    mrp->start_addr = malloc(mrp->count * mrp->size);
    if(mrp->start_addr == NULL) {
        return -1;
    }
    if(fread(mrp->start_addr, mrp->size, mrp->count, cp_file) != mrp->count) {
        return -1;
    }
    return 0;
}

int mini_cpr_restart_check(const char *name, int version) {
    FILE * fp;
    int found_version = version;
    printf("[mini-cpr] Info: Looking for checkpoints labeled %s with maximum version = %d\n", name,version);
    for (; found_version > -1; --found_version) {
        fp = mini_cpr_open_cp_file(name, found_version, "r");
        if(fp != NULL) {
            fclose(fp);
            break;
        }
    }
    if(found_version >= 0) {
      printf("[mini-cpr] Info: Found a checkpoint with version = %d\n", found_version);
    } else {
      printf("[mini-cpr] Info: Could not find a checkpoint with version <= %d\n", version);
    }
    return found_version;
}

int mini_cpr_restart(const char *name, int version) {
    printf("[mini-cpr] Info: Attempting restart from checkpoint labeled %s with version = %d\n", name, version);
    FILE * cp_file = mini_cpr_open_cp_file(name, version, "r");
    if(cp_file == NULL) {
        return -1;
    }
    while(1) {
        int cur_id;
        if(fread(&cur_id, sizeof(int), 1, cp_file) != 1) {
            if(feof(cp_file)) {
                break;
            } else {
                return -1;
            }
        }
        Mem_region_ptr cur_mrp = mini_cpr_rt_getMemregion(cur_id);
        if(cur_mrp == NULL) {
            return -1;
        }
        size_t cur_count;
        if(fread(&cur_count, sizeof(size_t), 1, cp_file) != 1) {
            if(feof(cp_file)) {
                break;
            } else {
                return -1;
            }
        }
        if(cur_count != cur_mrp->count) {
            return -1;
        }
        size_t cur_size;
        if(fread(&cur_size, sizeof(size_t), 1, cp_file) != 1) {
            if(feof(cp_file)) {
                break;
            } else {
                return -1;
            }
        }
        if(cur_size != cur_mrp->size) {
            return -1;
        }
        if(fread(cur_mrp->start_addr, cur_size, cur_count, cp_file) != cur_count) {
            if(feof(cp_file)) {
                break;
            } else {
                return -1;
            }
        }
    }
    printf("[mini-cpr] Info: Restart complete\n");
    return 0;
}
