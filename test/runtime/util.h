#ifndef UTIL_H
#define UTIL_H

#include "../../runtime/RuntimeInterface.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

int get_struct_id(int index) {
  return TA_NUM_RESERVED_IDS + index;
}

void check(void* addr, int id, int expected_count, int resolveStructs) {
  int id_result;
  size_t count_check;
  typeart_status status = typeart_get_type(addr, &id_result, &count_check);
  if (status == TA_OK) {
    if (resolveStructs) {
      // If the address corresponds to a struct, fetch the type of the first member
      while (id_result >= TA_NUM_RESERVED_IDS) {
        typeart_struct_layout struct_layout;
        typeart_resolve_type(id_result, &struct_layout);
        id_result   = struct_layout.member_types[0];
        count_check = struct_layout.count[0];
      }
    }

    if (id_result == id) {
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
  int id;
  size_t count_check;
  typeart_status status = typeart_get_type(addr, &id, &count_check);
  if (status == TA_OK) {
    if (id >= TA_NUM_RESERVED_IDS) {
      typeart_struct_layout struct_layout;
      typeart_resolve_type(id, &struct_layout);
      if (strcmp(typeart_get_type_name(id), struct_layout.name) != 0) {
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