#!/bin/bash

# shellcheck disable=SC2154
# shellcheck disable=SC1090

# RUN: chmod +x %s
# RUN: %s %wrapper-mpicxx | %filecheck %s --check-prefix=wcxx -DFCMPICXX=%mpicxx-compiler
# RUN: %s %wrapper-mpicc | %filecheck %s --check-prefix=wcc -DFCMPICC=%mpicc-compiler
# REQUIRES: mpi

source "$1" --version

# wcxx: TypeART-Toolchain:
# wcxx-NEXT: env OMPI_CXX={{.*}}clang++{{(-10|-11|-12|-13|-14)?}} [[FCMPICXX]]
# wcxx-NEXT: opt{{(-10|-11|-12|-13|-14)?}}
# wcxx-NEXT: llc{{(-10|-11|-12|-13|-14)?}}
# wcc: TypeART-Toolchain:
# wcc-NEXT: env OMPI_CC={{.*}}clang{{(-10|-11|-12|-13|-14)?}} [[FCMPICC]]
# wcc-NEXT: opt{{(-10|-11|-12|-13|-14)?}}
# wcc-NEXT: llc{{(-10|-11|-12|-13|-14)?}}
echo "TypeART-Toolchain:"
echo "$compiler"
echo "$opt_tool"
echo "$llc_tool"
