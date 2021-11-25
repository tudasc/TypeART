function(configure_typeart_script input output)
  cmake_parse_arguments(
    ARG "" "" "INSTALL_MODE;COMPILER;WITH_FILTER;APPLY_MODE" ${ARGN}
  )
  set(TYPEART_SAN_FLAGS "")

  set(TYPEART_PROJECT_DIR ${PROJECT_SOURCE_DIR})
  set(TYPEART_SCRIPT_DIR ${PROJECT_SOURCE_DIR}/scripts)

  set(TYPEART_MPI_INTERCEPT_LIB libinterceptor-rt.so)
  set(TYPEART_ANALYSIS_PLUGIN meminstfinderpass.so)
  set(TYPEART_PLUGIN typeartpass.so)

  if(ARG_INSTALL_MODE)
    set(TYPEART_INCLUDE_DIRS
        -I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
    )
    set(TYPEART_MPI_INTERCEPT_DIR
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TYPEART_RT_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_PASS_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_ANALYSIS_PASS_DIR
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TYPEART_PRELOAD_DIR
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TYPEART_BINARY_DIR -I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
    set(TYPEART_RELOCATABLE 1)
  else()
    set(TYPEART_INCLUDE_DIRS
        "-I${PROJECT_SOURCE_DIR}/lib/typelib -I${PROJECT_SOURCE_DIR}/lib/runtime -I${PROJECT_SOURCE_DIR}/lib/passes/typegen"
    )
    set(TYPEART_MPI_INTERCEPT_DIR ${CMAKE_BINARY_DIR}/lib/mpi_interceptor)
    set(TYPEART_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
    set(TYPEART_PASS_DIR ${CMAKE_BINARY_DIR}/lib/passes)
    set(TYPEART_ANALYSIS_PASS_DIR ${TYPEART_PASS_DIR}/analysis)
    set(TYPEART_PRELOAD_DIR ${CMAKE_BINARY_DIR}/lib/preload)
    set(TYPEART_BINARY_DIR ${CMAKE_BINARY_DIR})
    set(TYPEART_RELOCATABLE 0)
  endif()

  # TODO use generator in the scripts, even if target does not exist. How?
  set(TYPEART_MPI_TOOL "libtypeartMPITool${CMAKE_DEBUG_POSTFIX}.so")
  if(TARGET typeart::MPITool)
    set(TYPEART_HAS_MPI_TOOL 1)
  else()
    set(TYPEART_HAS_MPI_TOOL 0)
  endif()

  if(ARG_COMPILER)
    set(TYPEART_COMPILER ${ARG_COMPILER})
  endif()
  set(TYPEART_OPT ${OPT_EXEC})
  set(TYPEART_LLC ${LLC_EXEC})

  if(ARG_APPLY_MODE)
    set(TYPEART_RUN_SCRIPT 0)
  else()
    set(TYPEART_RUN_SCRIPT 1)
  endif()

  if(ARG_WITH_FILTER)
    set(TYPEART_CALLFILTER "-call-filter")
  endif()

  if(ENABLE_TSAN)
    target_tsan_flags(TYPEART_SAN_FLAGS)
  endif()

  if(ENABLE_ASAN)
    target_asan_flags(asan_flags)
    set(TYPEART_SAN_FLAGS ${TYPEART_SAN_FLAGS} ${asan_flags})
  endif()

  if(ENABLE_UBSAN)
    target_ubsan_flags(ubsan_flags)
    set(TYPEART_SAN_FLAGS ${TYPEART_SAN_FLAGS} ${ubsan_flags})
  endif()

  list(JOIN TYPEART_SAN_FLAGS " " TYPEART_SAN_FLAGS)

  target_generate_file(${input} ${output})
endfunction()

function(configure_typeart_coverage_script input output)
  find_program(
    LLVMCOV_COMMAND
    NAMES llvm-cov-10 llvm-cov
    HINTS ${LLVM_TOOLS_BINARY_DIR}
  )
  if(LLVMCOV_COMMAND)
    file(
      GENERATE
      OUTPUT ${output}
      CONTENT "#!/usr/bin/env bash\n\n${LLVMCOV_COMMAND} gcov \"$@\"\n"
      FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ
        WORLD_READ
    )
  else()
    message(WARNING "Need llvm-cov for coverage script")
  endif()
endfunction()

function(find_clang_progs target names default)
  find_program(
    target-prog
    NAMES ${names}
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH
  )

  if(NOT target-prog)
    set(${target}
        ${default}
        PARENT_SCOPE
    )
    message(
      STATUS
        "Did not find clang program ${names} in ${LLVM_TOOLS_BINARY_DIR}. Using def. value: ${default}"
    )
  else()
    set(${target}
        ${target-prog}
        PARENT_SCOPE
    )
  endif()
  unset(target-prog CACHE)
endfunction()

find_clang_progs(CLANG_EXEC "clang;clang-10" "clang")
find_clang_progs(CLANGCXX_EXEC "clang++;clang++-10" "clang++")
find_clang_progs(LLC_EXEC "llc;llc-10" "llc")
find_clang_progs(OPT_EXEC "opt;opt-10" "opt")

configure_typeart_coverage_script(llvm-gcov.sh.in llvm-gcov.sh)

configure_typeart_script(typeart-tmpl.sh.in run.sh)
configure_typeart_script(typeart-tmpl.sh.in apply.sh
  APPLY_MODE ON
)

configure_typeart_script(
  typeart-tmpl.sh.in typeart-run${CMAKE_DEBUG_POSTFIX}.sh
  INSTALL_MODE ON
)
configure_typeart_script(
  typeart-tmpl.sh.in typeart-apply${CMAKE_DEBUG_POSTFIX}.sh
  INSTALL_MODE ON
  APPLY_MODE ON
)

configure_typeart_script(
  typeart-wrapper.in typeart-clang${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${CLANG_EXEC}
)
configure_typeart_script(
  typeart-wrapper.in typeart-clang++${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${CLANGCXX_EXEC}
)

configure_typeart_script(
  typeart-wrapper.in typeart-clang-test
  COMPILER ${CLANG_EXEC}
)
configure_typeart_script(
  typeart-wrapper.in typeart-clang++-test
  COMPILER ${CLANGCXX_EXEC}
)

if(${MPI_C_FOUND})
  configure_typeart_script(
    typeart-wrapper.in typeart-mpicc${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "env OMPI_CC=${CLANG_EXEC} ${MPI_C_COMPILER}"
  )
  configure_typeart_script(
    typeart-wrapper.in typeart-mpicc-test
    WITH_FILTER ON
    COMPILER "env OMPI_CC=${CLANG_EXEC} ${MPI_C_COMPILER}"
  )
endif()

if(${MPI_CXX_FOUND})
  configure_typeart_script(
    typeart-wrapper.in typeart-mpic++${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "env OMPI_CXX=${CLANGCXX_EXEC} ${MPI_CXX_COMPILER}"
  )
  configure_typeart_script(
    typeart-wrapper.in typeart-mpic++-test
    WITH_FILTER ON
    COMPILER "env OMPI_CXX=${CLANGCXX_EXEC} ${MPI_CXX_COMPILER}"
  )
endif()

install(
  PROGRAMS
    $<$<BOOL:${INSTALL_UTIL_SCRIPTS}>:${CMAKE_CURRENT_BINARY_DIR}/typeart-run${CMAKE_DEBUG_POSTFIX}.sh>
    $<$<BOOL:${INSTALL_UTIL_SCRIPTS}>:${CMAKE_CURRENT_BINARY_DIR}/typeart-apply${CMAKE_DEBUG_POSTFIX}.sh>
    ${CMAKE_CURRENT_BINARY_DIR}/typeart-clang++${CMAKE_DEBUG_POSTFIX}
    ${CMAKE_CURRENT_BINARY_DIR}/typeart-clang${CMAKE_DEBUG_POSTFIX}
    $<$<AND:$<BOOL:${MPI_C_FOUND}>,$<BOOL:${ENABLE_MPI_WRAPPER}>>:${CMAKE_CURRENT_BINARY_DIR}/typeart-mpicc${CMAKE_DEBUG_POSTFIX}>
    $<$<AND:$<BOOL:${MPI_CXX_FOUND}>,$<BOOL:${ENABLE_MPI_WRAPPER}>>:${CMAKE_CURRENT_BINARY_DIR}/typeart-mpic++${CMAKE_DEBUG_POSTFIX}>
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
