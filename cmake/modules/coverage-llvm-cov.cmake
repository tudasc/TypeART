find_program(LLVM_PROFDATA_COMMAND
  NAMES llvm-profdata-10 llvm-profdata
  HINTS ${LLVM_TOOLS_BINARY_DIR}
)
find_program(LLVMCOV_COMMAND
  NAMES llvm-cov-10 llvm-cov
  HINTS ${LLVM_TOOLS_BINARY_DIR}
)

if(LLVM_PROFDATA_COMMAND-NOTFOUND OR LLVMCOV_COMMAND-NOTFOUND)
  message(WARNING "llvm-cov program stack needed for coverage.")
endif()

add_custom_target(
  cov-merge
  COMMAND ${LLVM_PROFDATA_COMMAND} merge -sparse -o code.pro *.profraw
  DEPENDS ${target}
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
)

add_custom_target(
  cov-all-report
  COMMAND ${LLVMCOV_COMMAND} report `cat -s ta-binaries.txt`
          --instr-profile=code.pro -ignore-filename-regex=${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
  DEPENDS cov-merge
)

add_custom_target(
  cov-clean
  COMMAND rm ta-binaries.txt
  COMMAND rm *.pro
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
)

add_custom_target(
  cov-all-clean
  COMMAND rm *
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
)

function(make_llvm_cov_target target)
  add_custom_target(
    cov-binary-list-${target}
    COMMAND ${CMAKE_COMMAND} -E echo "-object $<TARGET_FILE:${target}>" >> ta-binaries.txt
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
    DEPENDS ${target}
  )

  add_custom_target(
    cov-merge-${target}
    COMMAND ${LLVM_PROFDATA_COMMAND} merge -sparse -o code-${target}.pro *.profraw
    DEPENDS ${target}
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
  )

  add_custom_target(
    cov-report-${target}
    COMMAND ${LLVMCOV_COMMAND} report -object $<TARGET_FILE:${target}>
            --instr-profile=code-${target}.pro -ignore-filename-regex=${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
    DEPENDS ${target} cov-merge-${target}
  )

  add_dependencies(cov-all-report cov-binary-list-${target})
endfunction()
