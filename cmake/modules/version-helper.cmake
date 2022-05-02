cmake_minimum_required(VERSION 3.14)

find_package(Git QUIET)

if(EXISTS ${ROOT_DIR}/.git AND GIT_FOUND)
  execute_process(
    OUTPUT_VARIABLE TYPEART_GIT_REV
    COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
    WORKING_DIRECTORY ${ROOT_DIR}
  )
  string(SUBSTRING "${TYPEART_GIT_REV}" 0 10 TYPEART_GIT_REV)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} diff --quiet --exit-code
    RESULT_VARIABLE DIFF_STATUS
    WORKING_DIRECTORY ${ROOT_DIR}
  )
  if(DIFF_STATUS AND DIFF_STATUS EQUAL 1)
    string(APPEND TYPEART_GIT_REV "+")
  endif()
else()
  set(TYPEART_GIT_REV "N/A")
endif()

configure_file(Version.cpp.in ${OUTPUT_DIR}/Version.cpp)
