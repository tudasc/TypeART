#!/bin/bash

target=$1
pathToPlugin=${2:-build/lib}
pluginArgs=${3:-""}
pathToRT=${4:-build/runtime}
scriptDir=$(dirname "$0")

tmpDir="./"
tmpfile="$tmpDir"/"${target##*/}"
extension="${target##*.}"

rtDir="$( cd "$pathToRT" && pwd )"

echo -e Running on "$target" using plugin: "$plugin"

echo $pathToPlugin
echo $pathToRT

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

if [ -e "${tmpDir}/types.yaml" ]; then
    rm "${tmpDir}/types.yaml"
fi

$compiler -S -emit-llvm "$target" -o "$tmpfile".ll -I${scriptDir}/../typelib
opt -load ${pathToPlugin}/analysis/MemInstFinderPass.so -load ${pathToPlugin}/TypeArtPass.so -typeart "$pluginArgs"< "$tmpfile".ll -o "$tmpfile".ll > /dev/null
llc "$tmpfile".ll -o "$tmpfile".s
$compiler "$tmpfile".s -L"$pathToRT" -ltypeart -o "$tmpfile".o
echo -e Executing with runtime lib
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$rtDir" "$tmpfile".o
