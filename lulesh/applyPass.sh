#!/bin/bash

target=$1
outfile=$2
flags=$3

plugin="MustSupportPass"
pluginCommand="must"
pathToPlugin="$HOME/git/llvm-must-support/build/lib"
tmpDir=/tmp
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

echo -e Running on "$target" using plugin: "$plugin"

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

$compiler -S -emit-llvm "$flags" -O3 "$target" -o "$tmpfile".ll

opt -load "$pathToPlugin"/"$plugin".so -$pluginCommand -must-stats -o "${tmpfile}_opt".ll < "$tmpfile".ll > /dev/null
opt -o "${tmpfile}_opt".ll -O3 < "${tmpfile}_opt".ll > /dev/null
llc -o "$outfile" "$tmpfile".ll  -O3
