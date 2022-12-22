#!/bin/bash

# RUN: chmod +x %s
# RUN: %s %wrapper-ir-compare %wrapper-cc %wrapper-cc %S | %filecheck %s


function exists() {
  if [ -f "$1" ]; then
    echo 1
  else
    echo 0
  fi
}

cd "$4"

# CHECK: 0
# CHECK: 0
exists 12_ir_viewer_target_heap.ll-a
exists 12_ir_viewer_target_heap.ll-b

python $1 -s -a "$2" -b "$3" 12_ir_viewer_target.c -- -g
# CHECK: 1
# CHECK: 1
exists 12_ir_viewer_target_heap.ll-a
exists 12_ir_viewer_target_heap.ll-b

rm 12_ir_viewer_target_heap.ll-a 12_ir_viewer_target_heap.ll-b

# CHECK: 0
# CHECK: 0
exists 12_ir_viewer_target_stack.ll-a
exists 12_ir_viewer_target_stack.ll-b

python $1 -s -m stack -a "$2" -b "$3" 12_ir_viewer_target.c -- -g
# CHECK: 1
# CHECK: 1
exists 12_ir_viewer_target_stack.ll-a
exists 12_ir_viewer_target_stack.ll-b

rm 12_ir_viewer_target_stack.ll-a 12_ir_viewer_target_stack.ll-b
