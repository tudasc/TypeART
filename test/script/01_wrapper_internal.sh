#!/bin/bash

# shellcheck disable=SC2154

# RUN: chmod +x %s
# RUN: %s %wrapper-cxx | FileCheck %s --check-prefix=wcxx
# RUN: %s %wrapper-cc | FileCheck %s --check-prefix=wcc
# RUN: TYPEART_WRAPPER=OFF %s %wrapper-cc | FileCheck %s --check-prefix=wrapper-off

# RUN: %s %wrapper-cxx | FileCheck %s
# RUN: %s %wrapper-cc | FileCheck %s

source "$1" --version

# wcxx: TypeART-Toolchain:
# wcxx-NEXT: clang++{{(-10)?}}
# wcxx-NEXT: opt{{(-10)?}}
# wcxx-NEXT: llc{{(-10)?}}
# wcc: TypeART-Toolchain:
# wcc-NEXT: clang{{(-10)?}}
# wcc-NEXT: opt{{(-10)?}}
# wcc-NEXT: llc{{(-10)?}}
echo "TypeART-Toolchain:"
echo $compiler
echo $opt_tool
echo $llc_tool

# CHECK: 0
# wrapper-off: 1
is_wrapper_disabled
echo $?

# CHECK: 1
is_linking -o binary
echo $?

# CHECK: 1
is_linking main.o -o binary
echo $?

# CHECK: 0
is_linking -c
echo $?

# CHECK: 1
skip_typeart -E main.c
echo $?

function parse_check() {
  echo \
    "$found_src_file" "$found_obj_file" "$found_exe_file" \
    "$found_fpic" "$skip_typeart" "$typeart_to_asm" \
    "${optimize}"

  echo "${source_file:-es}" "${object_file:-eo}" "${asm_file:-ea}" "${exe_file:-eb}"
}

# CHECK: 1 0 0 1 0 0 -O0
# CHECK-NEXT: tool.c eo ea eb
parse_cmd_line -shared -fPIC tool.c -o libtool.so
parse_check

# CHECK: 1 1 0 0 0 0 -O1
# CHECK-NEXT: main.c main.o ea eb
parse_cmd_line -O1 -g -c -o main.o main.c
parse_check

# a linker call:
# CHECK: 0 0 0 0 0 0 -O0
# CHECK-NEXT: es eo ea eb
parse_cmd_line main.o -o binary
parse_check

# CHECK: 1 1 0 0 0 0 -O3
# CHECK-NEXT: lulesh.cc lulesh.o ea eb
# CHECK-NEXT: -DUSE_MPI=1 -g -I. -Wall
parse_cmd_line -DUSE_MPI=1 -g -I. -Wall -O3 -c -o lulesh.o lulesh.cc
parse_check
echo "${ta_more_args}"

# a linker call
# CHECK: 0 0 0 0 0 0 -O3
# CHECK-NEXT: es eo ea eb
# CHECK-NEXT: -DUSE_MPI=1 lulesh.o lulesh-comm.o lulesh-viz.o lulesh-util.o lulesh-init.o -g -lm lulesh2.0
parse_cmd_line -DUSE_MPI=1 lulesh.o lulesh-comm.o lulesh-viz.o lulesh-util.o lulesh-init.o -g -O3 -lm -o lulesh2.0
parse_check
echo "${ta_more_args}"
