#!/bin/bash

target=$1
plugin=${2:-MustSupportPass}
pluginCommand=${3:-must}
pathToPlugin=${4:-build/lib}
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

echo -e Running on "$target" using plugin: "$plugin"

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

$compiler -S -emit-llvm "$target" -o "$tmpfile".ll

opt -print-after-all -load "$pathToPlugin"/"$plugin".so -$pluginCommand < "$tmpfile".ll > /dev/null
