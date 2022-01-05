set(TYPEART_PROFILE_DIR ${CMAKE_BINARY_DIR}/profiles)
file(MAKE_DIRECTORY ${TYPEART_PROFILE_DIR})

if(NOT TYPEART_LLVM_CODE_COVERAGE AND TYPEART_CODE_COVERAGE)
  include(modules/coverage-gcovr)
  include(modules/coverage-lcov)
endif()

if(TYPEART_LLVM_CODE_COVERAGE)
  include(modules/coverage-llvm-cov)
endif()

function(typeart_target_coverage_options target)
  get_target_property(target_type ${target} TYPE)

  if (NOT TYPEART_LLVM_CODE_COVERAGE AND TYPEART_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PUBLIC
      -O0
      -g
      --coverage
    )

    target_link_options(${target} PUBLIC
      --coverage
    )

    if(NOT target_type STREQUAL "OBJECT_LIBRARY")
      typeart_target_lcov(${target})
    endif()
  endif ()
  if (TYPEART_LLVM_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
      typeart_target_llvm_cov(${target})
    endif()
  endif ()
endfunction()



