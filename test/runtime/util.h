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
    typeart_status status = typeart_get_type(addr, &type_info, &count_check);
    if (status == TA_OK) {
        if (resolveStructs) {
            // If the address corresponds to a struct, fetch the type of the first member
            while (type_info.kind == STRUCT) {

                typeart_struct_layout struct_layout;
                typeart_resolve_type(type_info.id, &struct_layout);
                type_info = struct_layout.member_types[0];
                count_check = struct_layout.count[0];
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
            case TA_UNKNOWN_ADDRESS:
                fprintf(stderr, "Error: Unknown address\n");
                break;
            case TA_BAD_ALIGNMENT:
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
    typeart_status status = typeart_get_type(addr, &type_info, &count_check);
    if (status == TA_OK) {
        if (type_info.kind == STRUCT) {
            typeart_struct_layout struct_layout;
            typeart_resolve_type(type_info.id, &struct_layout);
            if (strcmp(typeart_get_type_name(type_info.id), struct_layout.name) != 0) {
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
            case TA_UNKNOWN_ADDRESS:
                fprintf(stderr, "Error: Unknown address\n");
                break;
            case TA_BAD_ALIGNMENT:
                fprintf(stderr, "Error: Bad alignment\n");
                break;
            default:
                fprintf(stderr, "Error: Unexpected status: %d\n", status);
                break;
        }
    }
}

#endif