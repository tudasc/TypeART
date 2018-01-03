#!/bin/bash

target=$1
plugin=${2:-MustSupportPass}
pluginCommand=${3:-must}
pathToPlugin=${4:-build/lib}
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"

echo -e Running on "$target" using plugin: "$plugin"

clang -S -emit-llvm "$target" -o "$tmpfile".ll

opt -load "$pathToPlugin"/"$plugin".so -$pluginCommand < "$tmpfile".ll > /dev/null
