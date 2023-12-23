#!/bin/bash

# RUN: chmod +x %s
# RUN: %s %wrapper-ir-viewer %wrapper-cc %S %python-interp | %filecheck %s


function exists() {
  if [ -f "$1" ]; then
    echo 1
  else
    echo 0
  fi
}

cd "$3"

python_interp="$4"

# CHECK: 0
# CHECK: 0
# CHECK: 0
# CHECK: 0
exists 12_ir_viewer_target_base.ll
exists 12_ir_viewer_target_heap.ll
exists 12_ir_viewer_target_opt.ll
exists 12_ir_viewer_target_stack.ll

"$python_interp" $1 -s -w "$2" 12_ir_viewer_target.c -- -g
# CHECK: 1
# CHECK: 1
# CHECK: 1
# CHECK: 1
# CHECK: 1
exists 12_ir_viewer_target_base.ll
exists 12_ir_viewer_target_heap.ll
exists 12_ir_viewer_target_opt.ll
exists 12_ir_viewer_target_stack.ll
exists 12_ir_viewer_target-types-ir-viewer.yaml

"$python_interp" $1 -c 12_ir_viewer_target.c
# CHECK: 0
# CHECK: 0
# CHECK: 0
# CHECK: 0
# CHECK: 0
exists 12_ir_viewer_target_base.ll
exists 12_ir_viewer_target_heap.ll
exists 12_ir_viewer_target_opt.ll
exists 12_ir_viewer_target_stack.ll
exists 12_ir_viewer_target-types-ir-viewer.yaml
