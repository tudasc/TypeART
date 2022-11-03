// RUN: echo --- > types.yaml
// RUN: TYPEART_WRAPPER_EMIT_IR=1 %wrapper-cc -O1 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s
// RUN: cat 11_wrapper_emit_ir_heap.ll | %filecheck %s --check-prefix ir-out
// RUN: cat 11_wrapper_emit_ir_opt.ll | %filecheck %s --check-prefix ir-out
// RUN: cat 11_wrapper_emit_ir_stack.ll | %filecheck %s --check-prefix ir-out

#include "../../lib/runtime/CallbackInterface.h"

int main(int argc, char** argv) {
  __typeart_alloc((const void*)2, 7, 1);  // OK
  return 0;
}

// CHECK: [Trace] Alloc 0x2 7 float128 16 1

// ir-out: source_filename = "{{.*}}11_wrapper_emit_ir.c"
