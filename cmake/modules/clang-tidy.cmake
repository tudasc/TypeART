function(add_tidy_target target comment)
  macro(filter_dir dir_name_)
    foreach(source_file ${ARG_SOURCES})
      string(FIND ${source_file} ${dir_name_} EXCLUDE_FOUND)
      if(NOT ${EXCLUDE_FOUND} EQUAL -1)
        list(REMOVE_ITEM ARG_SOURCES ${source_file})
      endif()
    endforeach()
  endmacro()

  cmake_parse_arguments(ARG "" "" "SOURCES;EXCLUDES;OTHER" ${ARGN})

  foreach(exclude ${ARG_EXCLUDES})
    filter_dir(${exclude})
  endforeach()

  find_program(TIDY_COMMAND
    NAMES clang-tidy clang-tidy-12 clang-tidy-11 clang-tidy-10
    HINTS ${LLVM_TOOLS_BINARY_DIR}
  )
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
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, no clang-tidy command found."
    )
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

function(make_tidy_check name sources)
  add_tidy_target(tidy-run-on-${name}
    "Clang-tidy run on ${name} translation units"
    SOURCES ${sources}
    OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_tidy_fix_target(tidy-fix-on-${name}
    "Clang-tidy run with fixes on ${name} translation units"
    SOURCES ${sources}
    OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR}
          -checks=-*,modernize-*,llvm-namespace-comment,google-explicit-constructor,-modernize-use-trailing-return-type,-modernize-use-using
  )
endfunction()
