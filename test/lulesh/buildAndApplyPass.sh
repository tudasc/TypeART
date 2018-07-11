#!/bin/bash

target=$1
outfile=$2
flags=$3

plugin="MustSupportPass"
pluginCommand="must"
pathToPlugin="../../build/lib"
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

echo -e Running on "$target" using plugin: "$plugin"

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

$compiler -S -emit-llvm "$flags" "$target" -O3 -o "$tmpfile".ll
opt -load "$pathToPlugin"/analysis/meminstfinderpass.so -load "$pathToPlugin"/typeartpass.so -typeart -typeart-alloca -alloca-array-only=false  -o "${tmpfile}".ll < "$tmpfile".ll > /dev/null
#opt -o "${tmpfile}_opt".ll -O0 < "${tmpfile}".ll > /dev/null

llc -o "$outfile" "${tmpfile}_opt".ll  -O3
