if (TYPEART_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  find_program(GCOVR_COMMAND gcovr)
  if(GCOVR_COMMAND)
    add_custom_target(gcovr-report
      COMMAND ${GCOVR_COMMAND} -r ${PROJECT_SOURCE_DIR} -e ${PROJECT_BINARY_DIR} -s -j 4
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Make coverage report"
      USES_TERMINAL
    )
  else()
    add_custom_target(gcovr-report
      COMMAND ${CMAKE_COMMAND} -E echo "gcovr-report does nothing, no gcovr executable found.")
  endif()
endif()
