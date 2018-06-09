#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "../../runtime/RuntimeInterface.h"

void check(void* addr, int id, int expected_count, int resolveStructs) {
    must_type_info type_info;
    size_t count_check;
    lookup_result status = must_support_get_type(addr, &type_info, &count_check);
    if (status == SUCCESS) {
        if (resolveStructs) {
            // If the address corresponds to a struct, fetch the type of the first member
            while (type_info.kind == STRUCT) {
                size_t len;
                const must_type_info *types;
                const size_t *count;
                const size_t *offsets;
                size_t extent;
                must_support_resolve_type(type_info.id, &len, &types, &count, &offsets, &extent);
                type_info = types[0];
                count_check = count[0];
            }
        }

        if (type_info.id == id) {
            if (count_check != expected_count) {
                printf("Error: Count mismatch (%zu)\n", count_check);
            } else {
                printf("Ok\n");
            }
        } else {
            printf("Error: Type mismatch\n");
        }

    } else {
        switch (status) {
            case UNKNOWN_ADDRESS:
                printf("Error: Unknown address\n");
                break;
            case BAD_ALIGNMENT:
                printf("Error: Bad alignment\n");
                break;
            default:
                printf("Error: Unexpected status: %d\n", status);
                break;
        }
    }
}

void check_struct(void* addr, const char* name, int expected_count) {
    must_type_info type_info;
    size_t count_check;
    lookup_result status = must_support_get_type(addr, &type_info, &count_check);
    if (status == SUCCESS) {
        if (type_info.kind == STRUCT) {
            size_t len;
            const must_type_info *types;
            const size_t *counts;
            const size_t *offsets;
            size_t extent;
            must_support_resolve_type(type_info.id, &len, &types, &counts, &offsets, &extent);
            if (strcmp(must_support_get_type_name(type_info.id), name) != 0) {
                printf("Error: Name mismatch\n");
            } else if (expected_count != count_check) {
                printf("Error: Count mismatch (%zu)\n", count_check);
            } else {
                printf("Ok\n");
            }
        } else {
            printf("Error: Not a struct\n");
        }
    } else {
        switch (status) {
            case UNKNOWN_ADDRESS:
                printf("Error: Unknown address\n");
                break;
            case BAD_ALIGNMENT:
                printf("Error: Bad alignment\n");
                break;
            default:
                printf("Error: Unexpected status: %d\n", status);
                break;
        }
    }
}

#endif