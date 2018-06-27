#!/bin/bash

target=$1
pathToPlugin=${2:-build/lib}
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

if [ -e "/tmp/types.yaml" ]; then
    rm "/tmp/types.yaml"
fi

echo -e Running on "$target" using plugin: "$plugin"

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

$compiler -S -emit-llvm "$target" -o "$tmpfile".ll

opt -print-after-all -load ${pathToPlugin}/analysis/MemInstFinderPass.so -load ${pathToPlugin}/TypeArtPass.so -typeart -typeart-stats < "$tmpfile".ll > /dev/null
