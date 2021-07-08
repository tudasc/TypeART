#!/bin/bash

# RUN: chmod +x %s
# RUN: %s %t %S %wrapper-mpicc run-demo | FileCheck %s --check-prefix check-working
# RUN: %s %t %S %wrapper-mpicc run-demo_broken | FileCheck %s --check-prefix check-broken

# REQUIRES: mpicc

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

clean_up "$1"

# check-working-NOT: Error
# check-broken: Error
