function(target_project_coverage_options target)
  if (ENABLE_CODE_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${target} PUBLIC
      -O0 -g --coverage
    )
    target_link_options(${target} PUBLIC --coverage)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(${target} INTERFACE
        -O0 -g -fprofile-instr-generate -fcoverage-mapping
        )
      target_link_options(${target} INTERFACE -fprofile-instr-generate)
    endif ()
  endif()
endfunction()

include(coverage-gcovr)
include(coverage-llvm-cov)



