set(TYPEART_PROFILE_DIR ${CMAKE_BINARY_DIR}/profiles)

file(MAKE_DIRECTORY ${TYPEART_PROFILE_DIR})

add_custom_target(
  cov-merge
  COMMAND llvm-profdata merge -sparse -o code.pro *.profraw
  DEPENDS ${target}
  WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
  DEPENDS ${target}
)

add_custom_target(
  cov-all-report
  COMMAND echo ${SO_OBJECTS}
  COMMAND llvm-cov report `cat -s ${TYPEART_PROFILE_DIR}/ta-binaries.txt` --instr-profile=${TYPEART_PROFILE_DIR}/code.pro
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS cov-merge
)

add_custom_target(
  cov-clean
  COMMAND rm ${TYPEART_PROFILE_DIR}/ta-binaries.txt
  COMMAND rm ${TYPEART_PROFILE_DIR}/*.pro
)

add_custom_target(
  cov-all-clean
  COMMAND rm ${TYPEART_PROFILE_DIR}/*
)

function(make_llvm_cov_target target)
  add_custom_target(
    cov-binary-list-${target}
    COMMAND ${CMAKE_COMMAND} -E echo "-object $<TARGET_FILE:${target}>" >> ${TYPEART_PROFILE_DIR}/ta-binaries.txt
    DEPENDS ${target}
  )

  add_custom_target(
    cov-merge-${target}
    COMMAND llvm-profdata merge -sparse -o code-${target}.pro *.profraw
    DEPENDS ${target}
    WORKING_DIRECTORY ${TYPEART_PROFILE_DIR}
    DEPENDS ${target}
  )

  add_custom_target(
    cov-report-${target}
    COMMAND llvm-cov report -object $<TARGET_FILE:${target}> --instr-profile=${TYPEART_PROFILE_DIR}/code-${target}.pro
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${target} cov-merge-${target}
  )

  add_dependencies(cov-all-report cov-binary-list-${target})
endfunction()