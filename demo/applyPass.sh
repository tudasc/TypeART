#!/bin/bash

target=$1
plugin=${2:-MustSupportPass.so}
pluginArgs=${3:--must}
pathToPlugin=${4:-build/lib}
pathToRT=${5:-build/runtime}
mpi=${6:-1}
outfile=${7}

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

if [ -e /tmp/musttypes ]; then
  rm /tmp/musttypes
fi

OMPI_CC=$c_compiler OMPI_CXX=$cxx_compiler $compiler_wrapper -S -emit-llvm "$target" -o "$tmpfile".ll
opt -load "$pathToPlugin"/"$plugin" $pluginArgs < "$tmpfile".ll -o "$tmpfile".ll > /dev/null
llc "$tmpfile".ll -o "$tmpfile".s
OMPI_CC=$c_compiler OMPI_CXX=$cxx_compiler $compiler_wrapper "$tmpfile".s -L"$pathToRT" -lmustsupport -o "$outfile"
#echo -e Executing with runtime lib
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$rtDir "$tmpfile".o
