if (ENABLE_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  find_program(GCOVR_COMMAND gcovr)
  if(GCOVR_COMMAND)

    if(NOT FETCHCONTENT_BASE_DIR)
      set(GCOV_DIR_E _deps)
    else()
      set(GCOV_DIR_E ${FETCHCONTENT_BASE_DIR})
    endif()

    add_custom_target(gcovr-report
      COMMAND ${GCOVR_COMMAND} . -r ${PROJECT_SOURCE_DIR} -e ${GCOV_DIR_E} -e test -s -j 4
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Make coverage report"
      USES_TERMINAL
      )
  else()
    message(WARNING "Could not find gcovr executable.")
    add_custom_target(gcovr-report
      COMMAND ${CMAKE_COMMAND} -E echo "gcovr-report does nothing, no gcovr executable found.")
  endif()
endif()