set(TYPEART_PROFILE_DIR ${CMAKE_BINARY_DIR}/profiles)
file(MAKE_DIRECTORY ${TYPEART_PROFILE_DIR})

if(NOT ENABLE_LLVM_CODE_COVERAGE AND ENABLE_CODE_COVERAGE)
  include(coverage-gcovr)
  include(coverage-lcov)
endif()

if(ENABLE_LLVM_CODE_COVERAGE)
  include(coverage-llvm-cov)
endif()

function(target_project_coverage_options target)
  get_target_property(target_type ${target} TYPE)

  if (NOT ENABLE_LLVM_CODE_COVERAGE AND ENABLE_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PUBLIC
      -O0
      -g
      --coverage
    )

    target_link_options(${target} PUBLIC
      --coverage
    )

    if(NOT target_type STREQUAL "OBJECT_LIBRARY")
      make_lcov_target(${target})
    endif()
  endif ()
  if (ENABLE_LLVM_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target} PUBLIC
      -O0
      -g
      -fprofile-instr-generate
      -fcoverage-mapping
    )

    target_link_options(${target} PUBLIC
      -fprofile-instr-generate
    )

    if(NOT target_type STREQUAL "OBJECT_LIBRARY")
      make_llvm_cov_target(${target})
    endif()
  endif ()
endfunction()



