cmake_minimum_required(VERSION 3.5)

project(MPI_type_demo
  LANGUAGES C)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

message(STATUS "Searching here: " ${CMAKE_PREFIX_PATH})

# required to load typelib before runtime! 
# (cf. https://stackoverflow.com/a/50064639) 
find_package(typeart_typelib REQUIRED)
find_package(typeart_runtime REQUIRED)


find_package(typeart_llvm REQUIRED)
get_property(TYPEART_PASS_LOC TARGET typeart::typeartpass PROPERTY LOCATION)
get_property(TYPEART_MEMINST_PASS_LOC TARGET typeart::meminstfinderpass PROPERTY LOCATION)


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
  # need to link runtime for the instrumentation function symbols, otherwise it errors out:
#  target_link_libraries(${target}
#     typeart::runtime
#  )
  set_target_properties(${target} PROPERTIES
    COMPILE_FLAGS
      "-Xclang -load -Xclang ${TYPEART_MEMINST_PASS_LOC} -Xclang -load -Xclang ${TYPEART_PASS_LOC} -mllvm -alloca-array-only=false -mllvm -typeart-alloca"
  )
endfunction()

function(make_run_target target comment)
  add_custom_target(run-${target}
    COMMAND env LD_PRELOAD=${CMAKE_BINARY_DIR}/libtool.so ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${target}
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
target_link_libraries(tool
   typeart::runtime
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

