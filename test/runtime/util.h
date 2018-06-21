#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "../../runtime/RuntimeInterface.h"

int get_struct_id(int index) {
    return N_BUILTIN_TYPES + index;
}

void check(void* addr, int id, int expected_count, int resolveStructs) {
    typeart_type_info type_info;
    size_t count_check;
    lookup_result status = typeart_get_type(addr, &type_info, &count_check);
    if (status == SUCCESS) {
        if (resolveStructs) {
            // If the address corresponds to a struct, fetch the type of the first member
            while (type_info.kind == STRUCT) {
                size_t len;
                const typeart_type_info *types;
                const size_t *count;
                const size_t *offsets;
                size_t extent;
                typeart_resolve_type(type_info.id, &len, &types, &count, &offsets, &extent);
                type_info = types[0];
                count_check = count[0];
            }
        }

        if (type_info.id == id) {
            if (count_check != expected_count) {
                fprintf(stderr, "Error: Count mismatch (%zu)\n", count_check);
            } else {
                fprintf(stderr, "Ok\n");
            }
        } else {
            fprintf(stderr, "Error: Type mismatch\n");
        }

    } else {
        switch (status) {
            case UNKNOWN_ADDRESS:
                fprintf(stderr, "Error: Unknown address\n");
                break;
            case BAD_ALIGNMENT:
                fprintf(stderr, "Error: Bad alignment\n");
                break;
            default:
                fprintf(stderr, "Error: Unexpected status: %d\n", status);
                break;
        }
    }
}

void check_struct(void* addr, const char* name, int expected_count) {
    typeart_type_info type_info;
    size_t count_check;
    lookup_result status = typeart_get_type(addr, &type_info, &count_check);
    if (status == SUCCESS) {
        if (type_info.kind == STRUCT) {
            size_t len;
            const typeart_type_info *types;
            const size_t *counts;
            const size_t *offsets;
            size_t extent;
            typeart_resolve_type(type_info.id, &len, &types, &counts, &offsets, &extent);
            if (strcmp(typeart_get_type_name(type_info.id), name) != 0) {
                fprintf(stderr, "Error: Name mismatch\n");
            } else if (expected_count != count_check) {
                fprintf(stderr, "Error: Count mismatch (%zu)\n", count_check);
            } else {
                fprintf(stderr, "Ok\n");
            }
        } else {
            fprintf(stderr, "Error: Not a struct\n");
        }
    } else {
        switch (status) {
            case UNKNOWN_ADDRESS:
                fprintf(stderr, "Error: Unknown address\n");
                break;
            case BAD_ALIGNMENT:
                fprintf(stderr, "Error: Bad alignment\n");
                break;
            default:
                fprintf(stderr, "Error: Unexpected status: %d\n", status);
                break;
        }
    }
}

#endif