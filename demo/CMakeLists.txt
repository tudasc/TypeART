# FIXME the integration of all typeart libs/includes is cumbersome. 
# -- Needed: install_target of typeart for a common specified folder for everything

cmake_minimum_required(VERSION 3.5)

project(MPI_type_demo
  LANGUAGES C)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

find_library(TYPEART_LIBRARY
  NAMES typeart libtypeart
  PATHS "${CMAKE_SOURCE_DIR}/../build/runtime"
)

find_library(typeart_pass
  TypeArtPass.so
  PATHS "${CMAKE_SOURCE_DIR}/../build/lib"
)

find_library(typeart_pass_analysis
  MemInstFinderPass.so
  PATHS "${CMAKE_SOURCE_DIR}/../build/lib/analysis"
)

function(make_mpi_target target)
  target_link_libraries(${target} ${MPI_LIBRARIES})

  set_target_properties(${target} PROPERTIES
    COMPILE_FLAGS 
      ${MPI_C_COMPILE_FLAGS}
  )
  
  set_target_properties(${target} PROPERTIES
    LINK_FLAGS 
      ${MPI_C_LINK_FLAGS}
   )
  
  target_include_directories(${target} 
    SYSTEM
    PUBLIC
      ${MPI_C_INCLUDE_PATH}
  )
endfunction()

function(apply_pass target)
  if(NOT TYPEART_LIBRARY-NOTFOUND)
    target_link_libraries(${target}
      ${TYPEART_LIBRARY}
    )
  else()
    message(STATUS "Typeart runtime libs not found")
  endif()
  
  if(NOT typeart_pass-NOTFOUND AND NOT typeart_pass_analysis-NOTFOUND)
    set_target_properties(${target} PROPERTIES
      COMPILE_FLAGS
        "-Xclang -load -Xclang ${typeart_pass_analysis} -Xclang -load -Xclang ${typeart_pass}"
      )
  else()
    message(STATUS "Typeart LLVM Pass libs not found")
  endif()
endfunction()

function(make_run_target target comment)
  add_custom_target(run-${target}
    COMMAND env LD_PRELOAD=\"${CMAKE_BINARY_DIR}/libtool.so ${TYPEART_LIBRARY}\" ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${target}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "${comment}"
    USES_TERMINAL
  )
  add_dependencies(run-${target} ${target})
  add_dependencies(run-${target} tool)
endfunction()



add_library(tool SHARED 
  tool.c
)
make_mpi_target(tool)
#target_link_libraries(tool
#  ${TYPEART_LIBRARY}
#)
target_include_directories(tool PRIVATE
  "${CMAKE_SOURCE_DIR}/../runtime"
  "${CMAKE_SOURCE_DIR}/../typelib"
)


add_executable(demo
  01_struct_example.c
)
make_mpi_target(demo)
apply_pass(demo)
make_run_target(demo "Working demo")


add_executable(demo_broken
  02_broken_struct_example.c
)
make_mpi_target(demo_broken)
apply_pass(demo_broken)
make_run_target(demo_broken "Broken demo")

