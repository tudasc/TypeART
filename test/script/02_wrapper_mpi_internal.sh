#!/bin/bash

# RUN: chmod +x %s
# RUN: %s %wrapper-mpicxx | FileCheck %s --check-prefix=wcxx -DFCMPICXX=%mpicxx-compiler
# RUN: %s %wrapper-mpicc | FileCheck %s --check-prefix=wcc -DFCMPICC=%mpicc-compiler
# REQUIRES: mpi

source "$1" --version

# wcxx: TypeART-Toolchain:
# wcxx-NEXT: env OMPI_CXX={{.*}}clang++{{(-10)?}} [[FCMPICXX]]
# wcxx-NEXT: opt{{(-10)?}}
# wcxx-NEXT: llc{{(-10)?}}
# wcc: TypeART-Toolchain:
# wcc-NEXT: env OMPI_CC={{.*}}clang{{(-10)?}} [[FCMPICC]]
# wcc-NEXT: opt{{(-10)?}}
# wcc-NEXT: llc{{(-10)?}}
echo "TypeART-Toolchain:"
echo $compiler
echo $opt_tool
echo $llc_tool
