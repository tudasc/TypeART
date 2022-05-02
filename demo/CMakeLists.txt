cmake_minimum_required(VERSION 3.14)

project(MPI_type_demo LANGUAGES C)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

function(demo_make_mpi_target target)
  target_link_libraries(${target} PUBLIC MPI::MPI_C)
endfunction()

function(demo_make_run_target target comment)
  add_custom_target(
    run-${target}
    COMMAND env LD_PRELOAD=${CMAKE_BINARY_DIR}/libtool.so ${MPIEXEC}
            ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
            ${target}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "${comment}"
    USES_TERMINAL
  )
  add_dependencies(run-${target} ${target})
  add_dependencies(run-${target} tool)
endfunction()

add_library(tool SHARED tool.c)
demo_make_mpi_target(tool)

add_executable(demo 01_struct_example.c)
demo_make_mpi_target(demo)
demo_make_run_target(demo "Working demo")

add_executable(demo_broken 02_broken_struct_example.c)
demo_make_mpi_target(demo_broken)
demo_make_run_target(demo_broken "Broken demo")
