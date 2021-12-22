#!/bin/bash

# RUN: chmod +x %s
# RUN: %s %t %S %wrapper-mpicc run-demo | %filecheck %s --check-prefix check-working
# RUN: %s %t %S %wrapper-mpicc run-demo_broken | %filecheck %s --check-prefix check-broken

# REQUIRES: mpicc
# UNSUPPORTED: sanitizer

function clean_up() {
  if [ -d "$1" ]; then
    rm -rf "$1"
  fi
}

clean_up "$1"

TYPEART_WRAPPER=OFF cmake -B "$1" -S "$2"/../../demo -DCMAKE_C_COMPILER="$3"
cmake --build "$1" --target "$4"

# make sure "target" worked:
if [ $? -gt 0 ]; then
  clean_up "$1"
  exit 1
fi

clean_up "$1"

# check-working-NOT: [Demo] Error
# check-broken: [Demo] Error
