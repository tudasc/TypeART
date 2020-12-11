function(target_project_coverage_options target)
  if (ENABLE_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${target} INTERFACE
      -O0 -g --coverage
      )
    target_link_options(${target} INTERFACE --coverage)
  endif ()
  if (ENABLE_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target} INTERFACE
      -O0 -g -fprofile-instr-generate -fcoverage-mapping
      )
    target_link_options(${target} INTERFACE -fprofile-instr-generate)
  endif ()
endfunction()

set(TYPEART_PROFILE_DIR ${CMAKE_BINARY_DIR}/profiles)