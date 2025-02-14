// RUN: echo --- > typeart-types.yaml
// RUN: TYPEART_WRAPPER_EMIT_IR=1 %wrapper-cc -O1 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s
// RUN: cat 11_wrapper_emit_ir_heap.ll | %filecheck %s --check-prefix ir-out
// RUN: cat 11_wrapper_emit_ir_opt.ll | %filecheck %s --check-prefix ir-out
// RUN: cat 11_wrapper_emit_ir_stack.ll | %filecheck %s --check-prefix ir-out

#include "../../lib/runtime/CallbackInterface.h"
#include "TypeInterface.h"

int main(int argc, char** argv) {
  __typeart_alloc((const void*)2, TYPEART_FLOAT_128, 1);  // OK
  return 0;
}

// CHECK: [Trace] Alloc 0x2 24 long double 16 1

// ir-out: source_filename = "{{.*}}11_wrapper_emit_ir.c"
