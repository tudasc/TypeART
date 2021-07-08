macro(make_executable input output)
  # Executable permission: In CMAKE 3.19 we can simply use file(CHMOD) instead of file(copy) workaround
  configure_file(${input} ${CMAKE_CURRENT_BINARY_DIR}/tmp/${output} @ONLY)
  file(
    COPY ${CMAKE_CURRENT_BINARY_DIR}/tmp/${output}
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/tmp)
endmacro()

function(configure_typeart_script input output)
  cmake_parse_arguments(ARG "" "" "INSTALL_MODE;COMPILER;WITH_FILTER" ${ARGN})
  set(TYPEART_SAN_FLAGS "")

  set(TYPEART_PROJECT_DIR ${PROJECT_SOURCE_DIR})
  set(TYPEART_SCRIPT_DIR ${PROJECT_SOURCE_DIR}/scripts)

  set(TYPEART_MPI_INTERCEPT_LIB libinterceptor-rt.so)
  set(TYPEART_ANALYSIS_PLUGIN meminstfinderpass.so)
  set(TYPEART_PLUGIN typeartpass.so)

  if(ARG_INSTALL_MODE)
    set(TYPEART_INCLUDE_DIRS -I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
    set(TYPEART_MPI_INTERCEPT_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_RT_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_PASS_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_ANALYSIS_PASS_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_BINARY_DIR -I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
  else()
    set(TYPEART_INCLUDE_DIRS -I${PROJECT_SOURCE_DIR}/lib/typelib)
    set(TYPEART_MPI_INTERCEPT_DIR ${CMAKE_BINARY_DIR}/lib/mpi_interceptor)
    set(TYPEART_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
    set(TYPEART_PASS_DIR ${CMAKE_BINARY_DIR}/lib/passes)
    set(TYPEART_ANALYSIS_PASS_DIR ${TYPEART_PASS_DIR}/analysis)
    set(TYPEART_BINARY_DIR ${CMAKE_BINARY_DIR})
  endif()

  if(ARG_COMPILER)
    set(TYPEART_COMPILER ${ARG_COMPILER})
    set(TYPEART_OPT ${OPT_EXEC})
    set(TYPEART_LLC ${LLC_EXEC})
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

  if(ARG_INSTALL_MODE)
    configure_file(${input} ${output} @ONLY)
  else()
    # To get execute permission: create run.sh in the binary dir, and copy it to scripts folder with permission
    make_executable(${input} ${output})
  endif()
endfunction()

function(configure_typeart_coverage_script input output)
  find_program(LLVMCOV_COMMAND
    NAMES llvm-cov-10 llvm-cov
  )
  if(LLVMCOV_COMMAND)
    set(LLVM_COV_COMMAND ${LLVMCOV_COMMAND})
    make_executable(${input} ${output})
  else()
    message(WARNING "Need llvm-cov for coverage script")
  endif()
endfunction()

function(find_clang_progs target names default)
  unset(target-prog CACHE)
  find_program(target-prog
    NAMES ${names}
    PATHS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH
  )

  if (NOT target-prog)
    set(${target} ${default} PARENT_SCOPE)
    message(STATUS "Did not find clang program ${names} in ${LLVM_TOOLS_BINARY_DIR}. Using def. value: ${default}")
  else()
    set(${target} ${target-prog} PARENT_SCOPE)
  endif()
endfunction()


find_clang_progs(CLANG_EXEC "clang;clang-10" "clang")
find_clang_progs(CLANGCXX_EXEC "clang++;clang++-10" "clang++")
find_clang_progs(LLC_EXEC "llc;llc-10" "llc")
find_clang_progs(OPT_EXEC "opt;opt-10" "opt")

configure_typeart_coverage_script(llvm-gcov.sh.in llvm-gcov.sh)

configure_typeart_script(run.sh.in run.sh)
configure_typeart_script(apply.sh.in apply.sh)

configure_typeart_script(run.sh.in typeart-run.sh
  INSTALL_MODE ON
)
configure_typeart_script(apply.sh.in typeart-apply.sh
  INSTALL_MODE ON
)

configure_typeart_script(typeart-wrapper.in typeart-clang
  INSTALL_MODE ON
  COMPILER ${CLANG_EXEC}
)
configure_typeart_script(typeart-wrapper.in typeart-clang++
  INSTALL_MODE ON
  COMPILER ${CLANGCXX_EXEC}
)

configure_typeart_script(typeart-wrapper.in typeart-clang-test
  COMPILER ${CLANG_EXEC}
)
configure_typeart_script(typeart-wrapper.in typeart-clang++-test
  COMPILER ${CLANGCXX_EXEC}
)

if(${MPI_C_FOUND})
  configure_typeart_script(typeart-wrapper.in typeart-mpicc
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "env OMPI_CC=${CLANG_EXEC} ${MPI_C_COMPILER}"
  )
  configure_typeart_script(typeart-wrapper.in typeart-mpicc-test
    WITH_FILTER ON
    COMPILER "env OMPI_CC=${CLANG_EXEC} ${MPI_C_COMPILER}"
  )
endif()

if(${MPI_CXX_FOUND})
  configure_typeart_script(typeart-wrapper.in typeart-mpic++
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "env OMPI_CXX=${CLANGCXX_EXEC} ${MPI_CXX_COMPILER}"
  )
  configure_typeart_script(typeart-wrapper.in typeart-mpic++-test
    WITH_FILTER ON
    COMPILER "env OMPI_CXX=${CLANGCXX_EXEC} ${MPI_CXX_COMPILER}"
  )
endif()

install(
  PROGRAMS
    $<$<BOOL:${INSTALL_UTIL_SCRIPTS}>:${CMAKE_CURRENT_BINARY_DIR}/typeart-run.sh>
    $<$<BOOL:${INSTALL_UTIL_SCRIPTS}>:${CMAKE_CURRENT_BINARY_DIR}/typeart-apply.sh>
    ${CMAKE_CURRENT_BINARY_DIR}/typeart-clang++
    ${CMAKE_CURRENT_BINARY_DIR}/typeart-clang
    $<$<AND:$<BOOL:${MPI_C_FOUND}>,$<BOOL:${ENABLE_MPI_WRAPPER}>>:${CMAKE_CURRENT_BINARY_DIR}/typeart-mpicc>
    $<$<AND:$<BOOL:${MPI_CXX_FOUND}>,$<BOOL:${ENABLE_MPI_WRAPPER}>>:${CMAKE_CURRENT_BINARY_DIR}/typeart-mpic++>
  DESTINATION bin
)
