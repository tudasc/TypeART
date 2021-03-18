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
  flags="-Xclang -disable-O0-optnone -Iruntime/tycart "
else
  compiler=clang++
  flags="-std=c++14 -Xclang -disable-O0-optnone -Iruntime/tycart "
fi

function show_ir() {
# FIXME -OX as argument for opt causes passed to run twice..
echo "$compiler $flags -S -emit-llvm "$target" -o - | opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -load "$pathToPlugin"/typeartpass.so -typeart -typeart-alloca=true -typeart-stats -alloca-array-only=false -S 2>&1"
  $compiler $flags -S -emit-llvm "$target" -o - | opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -load "$pathToPlugin"/typeartpass.so -typeart  -typeart-stats -alloca-array-only=false -call-filter -S 2>&1
}

function show_ir_mem() {
# FIXME -OX as argument for opt causes passed to run twice..
  echo $compiler $flags -S -emit-llvm "$target" -o - | opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -mem-inst-finder

  $compiler $flags -S -emit-llvm "$target" -o - | opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -mem-inst-finder -alloca-array-only=false -call-filter -S 2>&1
}

show_ir
