if (TYPEART_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  find_program(TYPEART_GCOVR_EXEC gcovr)
  if(TYPEART_GCOVR_EXEC)
    add_custom_target(typeart-gcovr-report
      COMMAND ${TYPEART_GCOVR_EXEC} -r ${PROJECT_SOURCE_DIR} -e ${PROJECT_BINARY_DIR} -s -j 4
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Make coverage report with GCOVR tool."
      USES_TERMINAL
    )
  else()
    add_custom_target(typeart-gcovr-report
      COMMAND ${CMAKE_COMMAND} -E echo "gcovr-report does nothing, no gcovr executable found.")
  endif()
endif()
