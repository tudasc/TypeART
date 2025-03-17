// RUN: %run %s --typeart-stack-lifetime=true 2>&1 | %filecheck %s

#include "../../lib/runtime/RuntimeInterface.h"

#include <stdio.h>

void type_check(const void* addr) {
  typeart_status status;
  typeart_type_info info;
  status = typeart_get_type(addr, &info);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: %i %zu\n", info.type_id, info.count);
  }
}

void correct(int rank) {
  if (rank == 1) {
    // CHECK: Status OK: 13 9
    // CHECK: Status OK: 13 8
    // CHECK: Status OK: 13 7
    // CHECK: Status OK: 13 1
    int buffer[3][3] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    type_check(buffer);
    type_check(&buffer[0][1]);
    type_check(&buffer[0][2]);
    type_check(&buffer[2][2]);
  } else {
    // CHECK: Status OK: 13 3
    // CHECK: Status OK: 13 1
    int rcv[3] = {0, 1, 2};
    type_check(rcv);
    type_check(&rcv[2]);
  }
}

int main(void) {
  correct(1);
  correct(0);
  return 0;
}
