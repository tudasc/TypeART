
#include <stdio.h>
#include "../../runtime/RuntimeInterface.h"

void check(void* addr, int id) {
    must_type_info type_info;
    size_t count_check;
    lookup_result status = must_support_get_type(addr, &type_info, &count_check);
    if (status == SUCCESS) {
        // If the address corresponds to a struct, fetch the type of the first member
        while (type_info.kind == STRUCT) {
            size_t len;
            const must_type_info* types;
            const size_t* count;
            const size_t* offsets;
            size_t extent;
            must_support_resolve_type(type_info.id, &len, &types, &count, &offsets, &extent);
            type_info = types[0];
        }

        if (type_info.id == id) {
            printf("Ok\n");
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
            case WRONG_KIND:
                printf("Error: Wrong kind\n");
                break;
            default:
                printf("Error: Invalid status\n");
                break;
        }
    }
}