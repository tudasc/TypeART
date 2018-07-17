#!/bin/bash

target="$1"
outfile="$2"
flags="$3"
compiler="$4"

plugin="TypeArtPass"
pathToPlugin="../../build/lib"
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

echo "$compiler" -S -emit-llvm "$flags" "$target" -O3 -o "$tmpfile".ll
OMPI_CXX=clang++ mpic++ -DUSE_MPI=1 -S -emit-llvm $flags "$target" -o "$tmpfile".ll

opt -O3 -o "${tmpfile}".ll < "$tmpfile".ll > /dev/null
#echo opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -load "$pathToPlugin"/typeartpass.so -typeart -typeart-alloca -alloca-array-only=false -o "${tmpfile}".ll < "$tmpfile".ll > /dev/null
opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -load "$pathToPlugin"/typeartpass.so -typeart -typeart-alloca -alloca-array-only=false -o "${tmpfile}".ll < "$tmpfile".ll > /dev/null

#echo llc -O3 -filetype=obj -o "$outfile" "${tmpfile}_opt".ll
llc -filetype=obj -o "$outfile" "${tmpfile}_opt".ll

echo