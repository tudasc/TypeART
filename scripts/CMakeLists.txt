function(typeart_configure_script input output)
  cmake_parse_arguments(
    ARG "" "" "INSTALL_MODE;COMPILER;WITH_FILTER;APPLY_MODE" ${ARGN}
  )
  set(TYPEART_SAN_FLAGS "")

  set(TYPEART_PROJECT_DIR ${PROJECT_SOURCE_DIR})
  set(TYPEART_SCRIPT_DIR ${PROJECT_SOURCE_DIR}/scripts)

  if(ARG_INSTALL_MODE)
    set(TYPEART_INCLUDE_DIRS
        "-I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
    )
    set(TYPEART_MPI_INTERCEPT_DIR
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TYPEART_RT_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_PASS_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(TYPEART_ANALYSIS_PASS_DIR
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TYPEART_BINARY_DIR -I${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
    set(TYPEART_RELOCATABLE 1)
  else()
    set(TYPEART_INCLUDE_DIRS
        "-I${PROJECT_SOURCE_DIR}/lib/typelib -I${PROJECT_SOURCE_DIR}/lib/runtime -I${PROJECT_SOURCE_DIR}/lib/passes/typegen"
    )
    if(LIBRARY_OUTPUT_PATH)
      set(TYPEART_MPI_INTERCEPT_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_RT_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_PASS_DIR ${LIBRARY_OUTPUT_PATH})
      set(TYPEART_ANALYSIS_PASS_DIR ${LIBRARY_OUTPUT_PATH})
    else()
      set(TYPEART_MPI_INTERCEPT_DIR ${CMAKE_BINARY_DIR}/lib/mpi_interceptor)
      set(TYPEART_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
      set(TYPEART_PASS_DIR ${CMAKE_BINARY_DIR}/lib/passes)
      set(TYPEART_ANALYSIS_PASS_DIR ${TYPEART_PASS_DIR}/analysis)
    endif()

    if(EXECUTABLE_OUTPUT_PATH)
        set(TYPEART_BINARY_DIR ${EXECUTABLE_OUTPUT_PATH})
    else()
        set(TYPEART_BINARY_DIR ${CMAKE_BINARY_DIR})
    endif()

    set(TYPEART_RELOCATABLE 0)
  endif()

  # TODO use generator in the scripts, even if target does not exist. How?
  set(TYPEART_MPI_TOOL "typeartMPITool.so")
  if(TARGET typeart::MPITool)
    set(TYPEART_HAS_MPI_TOOL 1)
  else()
    set(TYPEART_HAS_MPI_TOOL 0)
  endif()

  if(ARG_COMPILER)
    set(TYPEART_COMPILER ${ARG_COMPILER})
  endif()

  set(TYPEART_OPT "${TYPEART_OPT_EXEC}")
  if(${LLVM_VERSION_MAJOR} VERSION_GREATER_EQUAL "13")
    set(TYPEART_OPT "${TYPEART_OPT} -enable-new-pm=0")
  endif()

  set(TYPEART_LLC "${TYPEART_LLC_EXEC}")

  if(ARG_APPLY_MODE)
    set(TYPEART_RUN_SCRIPT 0)
  else()
    set(TYPEART_RUN_SCRIPT 1)
  endif()

  if(ARG_WITH_FILTER)
    set(TYPEART_CALLFILTER "-typeart-call-filter")
  endif()

  if(TYPEART_TSAN)
    typeart_target_tsan_flags(TYPEART_SAN_FLAGS)
  endif()

  if(TYPEART_ASAN)
    typeart_target_asan_flags(asan_flags)
    set(TYPEART_SAN_FLAGS ${TYPEART_SAN_FLAGS} ${asan_flags})
  endif()

  if(TYPEART_UBSAN)
    typeart_target_ubsan_flags(ubsan_flags)
    set(TYPEART_SAN_FLAGS ${TYPEART_SAN_FLAGS} ${ubsan_flags})
  endif()

  list(JOIN TYPEART_SAN_FLAGS " " TYPEART_SAN_FLAGS)

  typeart_target_generate_file(${input} ${output})
endfunction()

function(typeart_configure_coverage_script input output)
  typeart_find_llvm_progs(TYPEART_LLVMCOV_COMMAND "llvm-cov-${LLVM_VERSION_MAJOR};llvm-cov")
  if(TYPEART_LLVMCOV_COMMAND)
    file(
      GENERATE
      OUTPUT ${output}
      CONTENT "#!/usr/bin/env bash\n\n${TYPEART_LLVMCOV_COMMAND} gcov \"$@\"\n"
      FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ
        WORLD_READ
    )
  else()
    message(WARNING "Need llvm-cov for coverage script")
  endif()
endfunction()

function(typeart_find_mpi_vendor_helper symbol ret_value)
  find_package(MPI)

  if(NOT MPI_FOUND)
    set(${ret_value} false)
    return()
  endif()

  set(MPI_TEST_CODE
   "#include <mpi.h>
    int main(void) {
      #if !defined(${symbol})
        an_illegal_stmt
      #endif
      return 0;
    }")

  set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_FLAGS "${MPI_C_COMPILE_FLAGS}")
  set(CMAKE_REQUIRED_DEFINITIONS "")
  set(CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")

  check_c_source_compiles("${MPI_TEST_CODE}" ${ret_value})
endfunction()

function(set_typeart_mpi_compiler)
  # Note on mpich & Intel: Intel may also define \"MPICH_NAME\"
  # in mpi.h, so check MPICH last
  list(APPEND l_vendor OPEN_MPI INTEL_MPI MPICH)
  list(APPEND l_symbol OPEN_MPI I_MPI_VERSION MPICH_NAME)
  list(APPEND l_env_c OMPI_CC I_MPI_CC MPICH_CC)
  list(APPEND l_env_cxx OMPI_CXX I_MPI_CXX MPICH_CXX)

  foreach(vendor symbol env_c env_cxx IN ZIP_LISTS l_vendor l_symbol l_env_c l_env_cxx)
    typeart_find_mpi_vendor_helper(${symbol} TYPEART_HAVE_${vendor})
    if(TYPEART_HAVE_${vendor})
      if(MPI_C_FOUND)
        set(TYPEART_MPICC
            "env ${env_c}=${TYPEART_CLANG_EXEC} ${MPI_C_COMPILER}"
            CACHE STRING "TypeART MPICC compiler command for scripts"
          )
        mark_as_advanced(TYPEART_MPICC)
      endif()
      if(MPI_CXX_FOUND)
        set(TYPEART_MPICXX
            "env ${env_cxx}=${TYPEART_CLANGCXX_EXEC} ${MPI_CXX_COMPILER}"
            CACHE STRING "TypeART MPICXX compiler command for scripts"
        )
        mark_as_advanced(TYPEART_MPICXX)
      endif()
      break()
    endif()
  endforeach()
endfunction()

typeart_configure_coverage_script(llvm-gcov.sh.in llvm-gcov.sh)

typeart_configure_script(typeart-tmpl.sh.in run.sh)
typeart_configure_script(typeart-tmpl.sh.in apply.sh
  APPLY_MODE ON
)

typeart_configure_script(
  typeart-tmpl.sh.in typeart-run${CMAKE_DEBUG_POSTFIX}.sh
  INSTALL_MODE ON
)
typeart_configure_script(
  typeart-tmpl.sh.in typeart-apply${CMAKE_DEBUG_POSTFIX}.sh
  INSTALL_MODE ON
  APPLY_MODE ON
)

typeart_configure_script(
  typeart-wrapper.in typeart-clang${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${TYPEART_CLANG_EXEC}
)
typeart_configure_script(
  typeart-wrapper.in typeart-clang++${CMAKE_DEBUG_POSTFIX}
  INSTALL_MODE ON
  COMPILER ${TYPEART_CLANGCXX_EXEC}
)

typeart_configure_script(
  typeart-wrapper.in typeart-clang-test
  COMPILER ${TYPEART_CLANG_EXEC}
)
typeart_configure_script(
  typeart-wrapper.in typeart-clang++-test
  COMPILER ${TYPEART_CLANGCXX_EXEC}
)

if(MPI_FOUND)
  set_typeart_mpi_compiler()
endif()

if(MPI_C_FOUND)
  typeart_configure_script(
    typeart-wrapper.in typeart-mpicc${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICC}"
  )
  typeart_configure_script(
    typeart-wrapper.in typeart-mpicc-test
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICC}"
  )
endif()

if(MPI_CXX_FOUND)
  typeart_configure_script(
    typeart-wrapper.in typeart-mpic++${CMAKE_DEBUG_POSTFIX}
    INSTALL_MODE ON
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICXX}"
  )
  typeart_configure_script(
    typeart-wrapper.in typeart-mpic++-test
    WITH_FILTER ON
    COMPILER "${TYPEART_MPICXX}"
  )
endif()

set(TYPEART_INSTALL_SCRIPT_CANDIDATE
  typeart-mpicc
  typeart-mpic++
  typeart-clang++
  typeart-clang
  typeart-run
  typeart-apply
  CACHE INTERNAL
  "Names of TypeART scripts which can be installed."
)

install(
  PROGRAMS
    $<$<BOOL:${TYPEART_INSTALL_UTIL_SCRIPTS}>:${CMAKE_CURRENT_BINARY_DIR}/typeart-run${CMAKE_DEBUG_POSTFIX}.sh>
    $<$<BOOL:${TYPEART_INSTALL_UTIL_SCRIPTS}>:${CMAKE_CURRENT_BINARY_DIR}/typeart-apply${CMAKE_DEBUG_POSTFIX}.sh>
    ${CMAKE_CURRENT_BINARY_DIR}/typeart-clang++${CMAKE_DEBUG_POSTFIX}
    ${CMAKE_CURRENT_BINARY_DIR}/typeart-clang${CMAKE_DEBUG_POSTFIX}
    $<$<AND:$<BOOL:${MPI_C_FOUND}>,$<BOOL:${TYPEART_MPI_WRAPPER}>>:${CMAKE_CURRENT_BINARY_DIR}/typeart-mpicc${CMAKE_DEBUG_POSTFIX}>
    $<$<AND:$<BOOL:${MPI_CXX_FOUND}>,$<BOOL:${TYPEART_MPI_WRAPPER}>>:${CMAKE_CURRENT_BINARY_DIR}/typeart-mpic++${CMAKE_DEBUG_POSTFIX}>
  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
