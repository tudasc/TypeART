function(add_format_target target comment)
  macro(filter_dir _dir_name_)
    foreach (SOURCE_FILE ${ALL_CXX_FILES})
      string(FIND ${SOURCE_FILE} ${_dir_name_} EXCLUDE_FOUND)
      if (NOT ${EXCLUDE_FOUND} EQUAL -1)
        list(REMOVE_ITEM ALL_CXX_FILES ${SOURCE_FILE})
      endif()
    endforeach()
  endmacro()

  cmake_parse_arguments(ARG "" "" "EXCLUDES;OTHER" ${ARGN})
  file(GLOB_RECURSE
    ALL_CXX_FILES
    lib/*.cpp
    lib/*.h
    runtime/*.cpp
    runtime/*.h
  )

  foreach(exclude ${ARG_EXCLUDES})
    filter_dir(${exclude})
  endforeach()

  find_program(FORMAT_COMMAND
               NAMES clang-format clang-format-5.0 clang-format-4.0 clang-format-3.8 clang-format-3.7)
  if(FORMAT_COMMAND)
    add_custom_target(${target}
      COMMAND ${FORMAT_COMMAND} -i -style=file ${ARG_OTHER} ${ARG_UNPARSED_ARGUMENTS}
              ${ALL_CXX_FILES}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMENT "${comment}"
      USES_TERMINAL
    )
  else()
    message(WARNING "Could not find clang-format executable.")
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, no clang-format found.")
  endif()
endfunction()


function(add_tidy_target target comment)
  macro(filter_dir _name_)
    foreach (SOURCE_FILE ${ARG_SOURCES})
      string(FIND ${SOURCE_FILE} ${_name_} EXCLUDE_FOUND)
      if (NOT ${EXCLUDE_FOUND} EQUAL -1)
        list(REMOVE_ITEM ARG_SOURCES ${SOURCE_FILE})
      endif()
    endforeach()
  endmacro()

  cmake_parse_arguments(ARG "" "" "SOURCES;EXCLUDES;OTHER" ${ARGN})

  foreach(exclude ${ARG_EXCLUDES})
    filter_dir(${exclude})
  endforeach()

  find_program(TIDY_COMMAND
               NAMES clang-tidy clang-tidy-5.0 clang-tidy-4.0 clang-tidy-3.8 clang-tidy-3.7)
  if(TIDY_COMMAND)
    add_custom_target(${target}
      COMMAND ${TIDY_COMMAND} -p ${CMAKE_BINARY_DIR}
              ${ARG_OTHER}
              ${ARG_UNPARSED_ARGUMENTS}
              ${ARG_SOURCES}
      COMMENT "${comment}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      USES_TERMINAL
    )
  else()
    message(WARNING "Could not find clang-tidy executable.")
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, no clang-tidy command found.")
  endif()
endfunction()

function(add_tidy_fix_target target comment)
  cmake_parse_arguments(ARG "" "" "SOURCES;EXCLUDES;OTHER" ${ARGN})
  add_tidy_target(${target} "${comment}"
    SOURCES ${ARG_SOURCES}
    EXCLUDES ${ARG_EXCLUDES}
    OTHER ${ARG_OTHER} -fix
  )
endfunction()
