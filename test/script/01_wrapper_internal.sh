#!/bin/bash

# shellcheck disable=SC2154
# shellcheck disable=SC1090

# RUN: chmod +x %s
# RUN: %s %wrapper-cxx | %filecheck %s --check-prefix=wcxx
# RUN: %s %wrapper-cc | %filecheck %s --check-prefix=wcc
# RUN: TYPEART_WRAPPER=OFF %s %wrapper-cc | %filecheck %s --check-prefix=wrapper-off

# RUN: %s %wrapper-cxx | %filecheck %s
# RUN: %s %wrapper-cc | %filecheck %s

# REQUIRES: legacywrapper

source "$1" --version

# wcxx: TypeART-Toolchain:
# wcxx-NEXT: clang++{{(-10|-11|-12|-13|-14)?}}
# wcxx-NEXT: opt{{(-10|-11|-12|-13|-14)?}}
# wcxx-NEXT: llc{{(-10|-11|-12|-13|-14)?}}
# wcc: TypeART-Toolchain:
# wcc-NEXT: clang{{(-10|-11|-12|-13|-14)?}}
# wcc-NEXT: opt{{(-10|-11|-12|-13|-14)?}}
# wcc-NEXT: llc{{(-10|-11|-12|-13|-14)?}}
echo "TypeART-Toolchain:"
echo "$typeart_compiler"
echo "$typeart_opt_tool"
echo "$typeart_llc_tool"

# CHECK: 0
# wrapper-off: 1
typeart_is_wrapper_disabled_fn
echo $?

# CHECK: 1
typeart_is_linking_fn -o binary
echo $?

# CHECK: 1
typeart_is_linking_fn main.o -o binary
echo $?

# CHECK: 0
typeart_is_linking_fn -c
echo $?

# CHECK: 1
typeart_skip_fn -E main.c
echo $?

function typeart_lit_parse_check_fn() {
  echo \
    "$typeart_found_src_file" "$typeart_found_obj_file" "$typeart_found_exe_file" \
    "$typeart_found_fpic" "$typeart_skip" "$typeart_to_asm" \
    "${typeart_optimize}"

  echo "${typeart_source_file:-es}" "${typeart_object_file:-eo}" "${typeart_asm_file:-ea}" "${typeart_exe_file:-eb}"
}

# CHECK: 1 0 0 1 0 0 -O0
# CHECK-NEXT: tool.c eo ea eb
typeart_parse_cmd_line_fn -shared  tool.c -o libtool.so
typeart_lit_parse_check_fn

# CHECK: 1 1 0 1 0 0 -O1
# CHECK-NEXT: main.c main.o ea eb
typeart_parse_cmd_line_fn -O1 -g -c -o main.o main.c
typeart_lit_parse_check_fn

# a linker call:
# CHECK: 0 0 0 1 0 0 -O0
# CHECK-NEXT: es eo ea eb
typeart_parse_cmd_line_fn main.o -o binary
typeart_lit_parse_check_fn

# CHECK: 1 1 0 1 0 0 -O3
# CHECK-NEXT: lulesh.cc lulesh.o ea eb
# CHECK-NEXT: -DUSE_MPI=1 -I. -Wall
typeart_parse_cmd_line_fn -DUSE_MPI=1 -I. -Wall -O3 -c -o lulesh.o lulesh.cc
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# a linker call:
# CHECK: 0 0 0 1 0 0 -O3
# CHECK-NEXT: es eo ea eb
# CHECK-NEXT: -DUSE_MPI=1 lulesh.o lulesh-comm.o lulesh-viz.o lulesh-util.o lulesh-init.o -lm lulesh2.0
typeart_parse_cmd_line_fn -DUSE_MPI=1 lulesh.o lulesh-comm.o lulesh-viz.o lulesh-util.o lulesh-init.o -O3 -lm -o lulesh2.0
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# CHECK: 1 1 0 1 0 0 -O2
# CHECK-NEXT: io_nonansi.c io_nonansi.o ea eb
# CHECK-NEXT: -I. -DFN -DFAST -DCONGRAD_TMP_VECTORS -DDSLASH_TMP_LINKS
typeart_parse_cmd_line_fn -c -I. -DFN -DFAST -DCONGRAD_TMP_VECTORS -DDSLASH_TMP_LINKS -g -O2 io_nonansi.c -o io_nonansi.o
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# CHECK: 1 1 0 1 0 0 -O2
# CHECK-NEXT: mgfparse.c mgfparse.o ea eb
# CHECK-NEXT: -DSPEC_MPI -DNDEBUG
typeart_parse_cmd_line_fn -DSPEC_MPI -DNDEBUG -g -O2 -c mgfparse.c -o mgfparse.o
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# CHECK: 1 1 0 1 0 0 -O2
# CHECK-NEXT: amg2013.c amg2013.o ea eb
# CHECK-NEXT: -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov -DHYPRE_USING_OPENMP -DTIMER_USE_MPI -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -fopenmp -DHYPRE_TIMING
typeart_parse_cmd_line_fn -o amg2013.o -c -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov -DHYPRE_USING_OPENMP -DTIMER_USE_MPI -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -O2 -g -fopenmp -DHYPRE_TIMING amg2013.c
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# a linker call:
# CHECK: 0 0 0 1 0 0 -O0
# CHECK-NEXT: es eo ea eb
# CHECK-NEXT: -L. -L../parcsr_ls -L../parcsr_mv -L../IJ_mv -L../seq_mv -L../sstruct_mv -L../struct_mv -L../krylov -L../utilities -lparcsr_ls -lparcsr_mv -lseq_mv -lsstruct_mv -lIJ_mv -lHYPRE_struct_mv -lkrylov -lHYPRE_utilities -lm -fopenmp
typeart_parse_cmd_line_fn -o amg2013 amg2013.o -L. -L../parcsr_ls -L../parcsr_mv -L../IJ_mv -L../seq_mv -L../sstruct_mv -L../struct_mv -L../krylov -L../utilities -lparcsr_ls -lparcsr_mv -lseq_mv -lsstruct_mv -lIJ_mv -lHYPRE_struct_mv -lkrylov -lHYPRE_utilities -lm -fopenmp
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# a linker call:
# CHECK: 0 0 1 1 0 0 -O0
# CHECK-NEXT: es eo ea libtool.so
# CHECK-NEXT: -fPIC -shared -Wl,-soname,libtool.so CMakeFiles/tool.dir/tool.c.o
typeart_linking=1 # This call would typically not be passed to typeart_parse_cmd_line_fn, linking is required for proper parsing.
typeart_parse_cmd_line_fn -fPIC -shared -Wl,-soname,libtool.so -o libtool.so CMakeFiles/tool.dir/tool.c.o
typeart_linking=0
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# CHECK: 1 1 0 1 0 0 -O0
# CHECK-NEXT: typeart/demo/tool.c CMakeFiles/tool.dir/tool.c.o ea eb
# CHECK-NEXT: -Dtool_EXPORTS -fPIC -MD -MT CMakeFiles/tool.dir/tool.c.o -MF CMakeFiles/tool.dir/tool.c.o.d
typeart_parse_cmd_line_fn -Dtool_EXPORTS  -fPIC -MD -MT CMakeFiles/tool.dir/tool.c.o -MF CMakeFiles/tool.dir/tool.c.o.d -o CMakeFiles/tool.dir/tool.c.o -c typeart/demo/tool.c
typeart_lit_parse_check_fn
echo "${typeart_wrapper_more_args}"

# CHECK: 0
typeart_global_env_var_init_fn
echo "${typeart_wrapper_emit_ir}"

# CHECK: 1
TYPEART_WRAPPER_EMIT_IR=1
typeart_global_env_var_init_fn
echo "${typeart_wrapper_emit_ir}"
