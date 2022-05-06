#!/bin/bash

# shellcheck disable=SC2154
# shellcheck disable=SC1090

# RUN: chmod +x %s
# RUN: %s %wrapper-cxx | %filecheck %s --check-prefix=wcxx
# RUN: %s %wrapper-cc | %filecheck %s --check-prefix=wcc
# RUN: TYPEART_WRAPPER=OFF %s %wrapper-cc | %filecheck %s --check-prefix=wrapper-off

# RUN: %s %wrapper-cxx | %filecheck %s
# RUN: %s %wrapper-cc | %filecheck %s

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
echo "$compiler"
echo "$opt_tool"
echo "$llc_tool"

# CHECK: 0
# wrapper-off: 1
is_wrapper_disabled
echo $?

# CHECK: 1
is_linking -o binary
echo $?

# CHECK: 1
is_linking main.o -o binary
echo $?

# CHECK: 0
is_linking -c
echo $?

# CHECK: 1
skip_typeart_compile -E main.c
echo $?

function parse_check() {
  echo \
    "$found_src_file" "$found_obj_file" "$found_exe_file" \
    "$found_fpic" "$skip_typeart" "$typeart_to_asm" \
    "${optimize}"

  echo "${source_file:-es}" "${object_file:-eo}" "${asm_file:-ea}" "${exe_file:-eb}"
}

# CHECK: 1 0 0 1 0 0 -O0
# CHECK-NEXT: tool.c eo ea eb
parse_cmd_line -shared -fPIC tool.c -o libtool.so
parse_check

# CHECK: 1 1 0 0 0 0 -O1
# CHECK-NEXT: main.c main.o ea eb
parse_cmd_line -O1 -g -c -o main.o main.c
parse_check

# a linker call:
# CHECK: 0 0 0 0 0 0 -O0
# CHECK-NEXT: es eo ea eb
parse_cmd_line main.o -o binary
parse_check

# CHECK: 1 1 0 0 0 0 -O3
# CHECK-NEXT: lulesh.cc lulesh.o ea eb
# CHECK-NEXT: -DUSE_MPI=1 -g -I. -Wall
parse_cmd_line -DUSE_MPI=1 -g -I. -Wall -O3 -c -o lulesh.o lulesh.cc
parse_check
echo "${ta_more_args}"

# a linker call:
# CHECK: 0 0 0 0 0 0 -O3
# CHECK-NEXT: es eo ea eb
# CHECK-NEXT: -DUSE_MPI=1 lulesh.o lulesh-comm.o lulesh-viz.o lulesh-util.o lulesh-init.o -g -lm lulesh2.0
parse_cmd_line -DUSE_MPI=1 lulesh.o lulesh-comm.o lulesh-viz.o lulesh-util.o lulesh-init.o -g -O3 -lm -o lulesh2.0
parse_check
echo "${ta_more_args}"

# CHECK: 1 1 0 0 0 0 -O2
# CHECK-NEXT: io_nonansi.c io_nonansi.o ea eb
# CHECK-NEXT: -I. -DFN -DFAST -DCONGRAD_TMP_VECTORS -DDSLASH_TMP_LINKS -g
parse_cmd_line -c -I. -DFN -DFAST -DCONGRAD_TMP_VECTORS -DDSLASH_TMP_LINKS -g -O2 io_nonansi.c -o io_nonansi.o
parse_check
echo "${ta_more_args}"

# CHECK: 1 1 0 0 0 0 -O2
# CHECK-NEXT: mgfparse.c mgfparse.o ea eb
# CHECK-NEXT: -DSPEC_MPI -DNDEBUG -g
parse_cmd_line -DSPEC_MPI -DNDEBUG -g -O2 -c mgfparse.c -o mgfparse.o
parse_check
echo "${ta_more_args}"

# CHECK: 1 1 0 0 0 0 -O2
# CHECK-NEXT: amg2013.c amg2013.o ea eb
# CHECK-NEXT: -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov -DHYPRE_USING_OPENMP -DTIMER_USE_MPI -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -g -fopenmp -DHYPRE_TIMING
parse_cmd_line -o amg2013.o -c -I.. -I../utilities -I../struct_mv -I../sstruct_mv -I../IJ_mv -I../seq_mv -I../parcsr_mv -I../parcsr_ls -I../krylov -DHYPRE_USING_OPENMP -DTIMER_USE_MPI -DHYPRE_LONG_LONG -DHYPRE_NO_GLOBAL_PARTITION -O2 -g -fopenmp -DHYPRE_TIMING amg2013.c
parse_check
echo "${ta_more_args}"

# a linker call:
# CHECK: 0 0 0 0 0 0 -O0
# CHECK-NEXT: es eo ea eb
# CHECK-NEXT: -L. -L../parcsr_ls -L../parcsr_mv -L../IJ_mv -L../seq_mv -L../sstruct_mv -L../struct_mv -L../krylov -L../utilities -lparcsr_ls -lparcsr_mv -lseq_mv -lsstruct_mv -lIJ_mv -lHYPRE_struct_mv -lkrylov -lHYPRE_utilities -lm -fopenmp
parse_cmd_line -o amg2013 amg2013.o -L. -L../parcsr_ls -L../parcsr_mv -L../IJ_mv -L../seq_mv -L../sstruct_mv -L../struct_mv -L../krylov -L../utilities -lparcsr_ls -lparcsr_mv -lseq_mv -lsstruct_mv -lIJ_mv -lHYPRE_struct_mv -lkrylov -lHYPRE_utilities -lm -fopenmp
parse_check
echo "${ta_more_args}"

# a linker call:
# CHECK: 0 0 1 1 0 0 -O0
# CHECK-NEXT: es eo ea libtool.so
# CHECK-NEXT: -fPIC -shared -Wl,-soname,libtool.so CMakeFiles/tool.dir/tool.c.o
linking=1 # This call would typically not be passed to parse_cmd_line, linking is required for proper parsing.
parse_cmd_line -fPIC -shared -Wl,-soname,libtool.so -o libtool.so CMakeFiles/tool.dir/tool.c.o
linking=0
parse_check
echo "${ta_more_args}"

# CHECK: 1 1 0 1 0 0 -O0
# CHECK-NEXT: typeart/demo/tool.c CMakeFiles/tool.dir/tool.c.o ea eb
# CHECK-NEXT: -Dtool_EXPORTS -fPIC -MD -MT CMakeFiles/tool.dir/tool.c.o -MF CMakeFiles/tool.dir/tool.c.o.d
parse_cmd_line -Dtool_EXPORTS  -fPIC -MD -MT CMakeFiles/tool.dir/tool.c.o -MF CMakeFiles/tool.dir/tool.c.o.d -o CMakeFiles/tool.dir/tool.c.o -c typeart/demo/tool.c
parse_check
echo "${ta_more_args}"
