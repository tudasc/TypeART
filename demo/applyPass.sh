#!/bin/bash

target=$1
pathToPlugin=${2:-../build/lib}
pathToRT=${3:-../build/runtime}
mpi=${4:-1}
outfile=${5}

tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

rtDir="$( cd "$pathToRT" && pwd )"

echo -e Running on "$target" using plugin: "$plugin"

c_compiler=clang
cxx_compiler=clang++

c_compiler_wrapper=$c_compiler
cxx_compiler_wrapper=$cxx_compiler

if [ $mpi == 1 ]; then
  c_compiler_wrapper=mpicc
  cxx_compiler_wrapper=mpic++
fi

if [ $extension == "c" ]; then
  compiler=$c_compiler
  compiler_wrapper=$c_compiler_wrapper
else
  compiler=$cxx_compiler
  compiler_wrapper=$cxx_compiler_wrapper
fi

if [ -e "types.yaml" ]; then
  rm "types.yaml"
fi

OMPI_CC=$c_compiler OMPI_CXX=$cxx_compiler $compiler_wrapper -S -emit-llvm "$target" -o "$tmpfile".ll
opt -load "${pathToPlugin}/analysis/meminstfinderpass.so" -load "${pathToPlugin}/typeartpass.so" -typeart -typeart-alloca < "$tmpfile".ll -S -o "$tmpfile".ll
llc "$tmpfile".ll -o "$tmpfile".s
OMPI_CC=$c_compiler OMPI_CXX=$cxx_compiler $compiler_wrapper "$tmpfile".s -L"$pathToRT" -ltypeart -o "$outfile"
