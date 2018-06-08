#!/bin/bash

target=$1
pathToPlugin=${2:-build/lib}
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

if [ -e "/tmp/musttypes" ]; then
    rm "/tmp/musttypes"
fi

echo -e Running on "$target" using plugin: "$plugin"

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

$compiler -S -emit-llvm "$target" -o "$tmpfile".ll

opt -print-after-all -load ${pathToPlugin}/analysis/MemInstFinderPass.so -load ${pathToPlugin}/MustSupportPass.so -must -must-stats < "$tmpfile".ll > /dev/null
