typeart_find_llvm_progs(TYPEART_LLVM_PROFDATA_COMMAND "llvm-profdata-${LLVM_VERSION_MAJOR};llvm-profdata" false)
typeart_find_llvm_progs(TYPEART_LLVMCOV_COMMAND "llvm-cov-${LLVM_VERSION_MAJOR};llvm-cov" false)

if(NOT TYPEART_LLVM_PROFDATA_COMMAND OR NOT TYPEART_LLVMCOV_COMMAND)
  message(WARNING "llvm-cov and llvm-profdata programs needed for coverage.")
endif()

add_custom_target(
  cov-merge
  COMMAND ${TYPEART_LLVM_PROFDATA_COMMAND} merge -sparse -o code.pro *.profraw
  DEPENDS ${target}
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
)

add_custom_target(
  cov-all-report
  COMMAND ${TYPEART_LLVMCOV_COMMAND} report `cat -s ta-binaries.txt`
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

function(typeart_target_llvm_cov target)
  add_custom_target(
    cov-binary-list-${target}
    COMMAND ${CMAKE_COMMAND} -E echo "-object $<TARGET_FILE:${target}>" >> ta-binaries.txt
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
    DEPENDS ${target}
  )

  add_custom_target(
    cov-merge-${target}
    COMMAND ${TYPEART_LLVM_PROFDATA_COMMAND} merge -sparse -o code-${target}.pro *.profraw
    DEPENDS ${target}
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
  )

  add_custom_target(
    cov-report-${target}
    COMMAND ${TYPEART_LLVMCOV_COMMAND} report -object $<TARGET_FILE:${target}>
            --instr-profile=code-${target}.pro -ignore-filename-regex=${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
    DEPENDS ${target} cov-merge-${target}
  )

  add_dependencies(cov-all-report cov-binary-list-${target})
endfunction()
