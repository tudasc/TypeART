#!/bin/bash

# shellcheck disable=SC2154
# shellcheck disable=SC1090

# RUN: chmod +x %s
# RUN: %s %wrapper-cc | %filecheck %s
# REQUIRES: cuda

source "$1" --version

function check_cuda() {
  echo C:"$typeart_found_cuda"
  typeart_found_cuda=0
  unset typeart_source_file
}

# CHECK: C:1
typeart_parse_cuda_cmd_line_fn file.cu
check_cuda

# CHECK-NEXT: C:1
typeart_parse_cuda_cmd_line_fn -x cuda
check_cuda

# CHECK-NEXT: C:0
typeart_parse_cuda_cmd_line_fn
check_cuda

# CHECK: C:file.cu
typeart_parse_cmd_line_fn file.cu
echo C:"$typeart_source_file"
