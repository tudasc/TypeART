#!/bin/bash

target=$1
pathToPlugin=${2:-build/lib}
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

if [ -e "types.yaml" ]; then
    rm "types.yaml"
fi

echo -e Running on "$target" using plugin: "$plugin"

if [ $extension == "c" ]; then
  compiler=clang
  flags="-Xclang -disable-O0-optnone "
else
  compiler=clang++
  flags="-std=c++14 -Xclang -disable-O0-optnone "
fi

function show_ir() {
# FIXME -OX as argument for opt causes passed to run twice..
  $compiler $flags -S -emit-llvm "$target" -o - | opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -load "$pathToPlugin"/typeartpass.so -typeart -typeart-alloca -typeart-stats -S 2>&1
}

function show_ir_mem() {
# FIXME -OX as argument for opt causes passed to run twice..
  $compiler $flags -S -emit-llvm "$target" -o - | opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -mem-inst-finder -S 2>&1
}

show_ir
