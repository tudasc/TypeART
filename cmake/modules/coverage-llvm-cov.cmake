typeart_find_llvm_progs(TYPEART_LLVM_PROFDATA_EXEC "llvm-profdata-${LLVM_VERSION_MAJOR};llvm-profdata")
typeart_find_llvm_progs(TYPEART_LLVMCOV_EXEC "llvm-cov-${LLVM_VERSION_MAJOR};llvm-cov")

if(NOT TYPEART_LLVM_PROFDATA_EXEC OR NOT TYPEART_LLVMCOV_EXEC)
  message(WARNING "llvm-cov and llvm-profdata programs needed for coverage.")
endif()

add_custom_target(
  typeart-cov-merge
  COMMAND ${TYPEART_LLVM_PROFDATA_EXEC} merge -sparse -o code.pro *.profraw
  DEPENDS ${target}
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
)

add_custom_target(
  typeart-cov-all-report
  COMMAND ${TYPEART_LLVMCOV_EXEC} report `cat -s typeart-binaries.txt`
          --instr-profile=code.pro -ignore-filename-regex=${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
  DEPENDS typeart-cov-merge
)

add_custom_target(
  typeart-cov-clean
  COMMAND rm typeart-binaries.txt
  COMMAND rm *.pro
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
)

add_custom_target(
  typeart-cov-all-clean
  COMMAND rm *
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
)

function(typeart_target_llvm_cov target)
  add_custom_target(
    typeart-cov-binary-list-${target}
    COMMAND ${CMAKE_COMMAND} -E echo "-object $<TARGET_FILE:${target}>" >> typeart-binaries.txt
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
    DEPENDS ${target}
  )

  add_custom_target(
    typeart-cov-merge-${target}
    COMMAND ${TYPEART_LLVM_PROFDATA_EXEC} merge -sparse -o code-${target}.pro *.profraw
    DEPENDS ${target}
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
  )

  add_custom_target(
    typeart-cov-report-${target}
    COMMAND ${TYPEART_LLVMCOV_EXEC} report -object $<TARGET_FILE:${target}>
            --instr-profile=code-${target}.pro -ignore-filename-regex=${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
    DEPENDS ${target} typeart-cov-merge-${target}
  )

  add_dependencies(typeart-cov-all-report typeart-cov-binary-list-${target})
endfunction()
