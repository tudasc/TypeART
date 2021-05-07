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
  cmake_parse_arguments(ARG "" "" "INSTALL_MODE" ${ARGN})
  set(TYPEART_SAN_FLAGS "")

  set(TYPEART_PROJECT_DIR ${PROJECT_SOURCE_DIR})
  set(TYPEART_BINARY_DIR ${CMAKE_BINARY_DIR})
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
  else()
    set(TYPEART_INCLUDE_DIRS -I${PROJECT_SOURCE_DIR}/lib/typelib)
    set(TYPEART_MPI_INTERCEPT_DIR ${CMAKE_BINARY_DIR}/lib/mpi_interceptor)
    set(TYPEART_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
    set(TYPEART_PASS_DIR ${CMAKE_BINARY_DIR}/lib/passes)
    set(TYPEART_ANALYSIS_PASS_DIR ${TYPEART_PASS_DIR}/analysis)
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

configure_typeart_script(run.sh.in run.sh)
configure_typeart_script(apply.sh.in apply.sh)

configure_typeart_script(run.sh.in run-typeart.sh
  INSTALL_MODE ON)
configure_typeart_script(apply.sh.in apply-typeart.sh
  INSTALL_MODE ON)

configure_typeart_coverage_script(llvm-gcov.sh.in llvm-gcov.sh)

#configure_file(clang-typeart.in clang-typeart)
#configure_file(clang-typeart++.in clang-typeart++)

install(
  PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/run-typeart.sh
    ${CMAKE_CURRENT_BINARY_DIR}/apply-typeart.sh
  DESTINATION bin
)

#install(
#  PROGRAMS
#    ${CMAKE_CURRENT_BINARY_DIR}/clang-typeart
#    ${CMAKE_CURRENT_BINARY_DIR}/clang-typeart++
#  DESTINATION bin
#)
