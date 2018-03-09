#!/bin/bash

target=$1
plugin=${2:-MustSupportPass.so}
pluginArgs=${3:--must}
pathToPlugin=${4:-build/lib}
pathToRT=${5:-build/runtime}
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

rtDir="$( cd "$pathToRT" && pwd )"

echo -e Running on "$target" using plugin: "$plugin"

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

$compiler -S -emit-llvm "$target" -o "$tmpfile".ll
opt -load "$pathToPlugin"/"$plugin" $pluginArgs < "$tmpfile".ll -o "$tmpfile".ll &> /dev/null
llc "$tmpfile".ll -o "$tmpfile".s
clang "$tmpfile".s -L"$pathToRT" -lmustsupport -o "$tmpfile".o
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$rtDir "$tmpfile".o
