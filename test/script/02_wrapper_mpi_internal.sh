#!/bin/bash

# shellcheck disable=SC2154
# shellcheck disable=SC1090

# RUN: chmod +x %s
# RUN: %s %wrapper-mpicxx | %filecheck %s --check-prefix=wcxx -DFCMPICXX=%mpicxx-compiler
# RUN: %s %wrapper-mpicc | %filecheck %s --check-prefix=wcc -DFCMPICC=%mpicc-compiler
# REQUIRES: mpi && legacywrapper

source "$1" --version

# wcxx: TypeART-Toolchain:
# wcxx-NEXT: env OMPI_CXX={{.*}}clang++{{(-14|-18)?}} [[FCMPICXX]]
# wcxx-NEXT: opt{{(-14|-18)?}}
# wcxx-NEXT: llc{{(-14|-18)?}}
# wcc: TypeART-Toolchain:
# wcc-NEXT: env OMPI_CC={{.*}}clang{{(-14|-18)?}} [[FCMPICC]]
# wcc-NEXT: opt{{(-14|-18)?}}
# wcc-NEXT: llc{{(-14|-18)?}}
echo "TypeART-Toolchain:"
echo "$typeart_compiler"
echo "$typeart_opt_tool"
echo "$typeart_llc_tool"
