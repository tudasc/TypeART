#!/bin/bash

# RUN: chmod +x %s
# RUN: %s %t %S %wrapper-mpicc run-demo | %filecheck %s --check-prefix check-working
# RUN: %s %t %S %wrapper-mpicc run-demo_broken | %filecheck %s --check-prefix check-broken

# RUN: %s %t %S %wrapper-mpicc runtoy | %filecheck %s --check-prefix check-toy

# REQUIRES: mpicc
# UNSUPPORTED: sanitizer

function clean_up() {
  if [ -d "$1" ]; then
    rm -rf "$1"
  fi
}

# Copy demo to tmp folder for tests:
clean_up "$1"
cp -R "$2"/../../demo "$1"/
cd "$1" || exit 1

make clean
MPICC="$3" make "$4"

# make sure "target" worked:
if [ $? -gt 0 ]; then
  clean_up "$1"
  exit 1
fi

clean_up "$1"

# check-working-NOT: [Demo] Error
# check-broken: [Demo] Error
# check-toy-NOT: [Demo] Toy Error
