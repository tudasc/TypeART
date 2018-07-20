find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}") 

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(clang-tidy)
include(clang-format)
include(llvm-util)
include(log-util)

set(LOG_LEVEL 3 CACHE STRING "Granularity of logger. 3 ist most verbose, 0 is least.")
option(MPI_LOGGER "Whether the logger should use MPI." OFF)
option(MPI_INTERCEPT_LIB "Build MPI interceptor library, requires wrap.py generator file." OFF)
option(SOFTCOUNTERS "Enable software tracking of #tracked addrs. / #distinct checks / etc." OFF)

if(MPI_LOGGER)
  find_package(MPI REQUIRED)
endif()

if(NOT CMAKE_BUILD_TYPE)
# set default build type
  set(CMAKE_BUILD_TYPE Debug)
endif()


function(target_project_compile_options target)
  cmake_parse_arguments(ARG "" "" "PRIVATE_FLAGS;PUBLIC_FLAGS" ${ARGN})

  target_compile_options(${target} PRIVATE
    -Wall -Wextra -pedantic
    -Wunreachable-code -Wwrite-strings
    -Wpointer-arith -Wcast-align
    -Wcast-qual -Wno-unused-parameter
  )

  if(ARG_PRIVATE_FLAGS)
    target_compile_options(${target} PRIVATE
      "${ARG_PRIVATE_FLAGS}"
    )
  endif()

  if(ARG_PUBLIC_FLAGS)
    target_compile_options(${target} PUBLIC
      "${ARG_PUBLIC_FLAGS}"
    )
  endif()
endfunction()

function(target_project_compile_definitions target)
  cmake_parse_arguments(ARG "" "" "PRIVATE_DEFS;PUBLIC_DEFS" ${ARGN})

  target_compile_definitions(${target} PRIVATE
    LOG_LEVEL=${LOG_LEVEL}
  )

  if(ARG_PRIVATE_DEFS)
    target_compile_definitions(${target} PRIVATE
      "${ARG_PRIVATE_DEFS}"
    )
  endif()

  if(ARG_PUBLIC_DEFS)
    target_compile_definitions(${target} PUBLIC
      "${ARG_PUBLIC_DEFS}"
    )
  endif()
endfunction()
