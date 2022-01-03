#!/bin/bash
#
# TypeART library
#
# Copyright (c) 2017-2022 TypeART Authors
# Distributed under the BSD 3-Clause license.
# (See accompanying file LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
#
# Project home: https://github.com/tudasc/TypeART
#
# SPDX-License-Identifier: BSD-3-Clause
#
#

target=$1
extension="${target##*.}"

if [ $extension == "c" ]; then
  compiler=clang
else
  compiler=clang++
fi

echo "|----------------------------------------------------------|"
echo "|------------------------Source----------------------------|"
echo "|----------------------------------------------------------|"
cat "$target"
echo "|----------------------------------------------------------|"
echo "|--------------------------LL------------------------------|"
echo "|----------------------------------------------------------|"
$compiler -S -emit-llvm -fno-discard-value-names "$target" -o -
