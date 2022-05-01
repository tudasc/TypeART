function(typeart_add_tidy_target target comment)
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

  typeart_find_llvm_progs(TYPEART_CLANG_TIDY_EXEC
    "clang-tidy-${LLVM_VERSION_MAJOR};clang-tidy"
  )

  if(TYPEART_CLANG_TIDY_EXEC)
    add_custom_target(${target}
      COMMAND ${TYPEART_CLANG_TIDY_EXEC} -p ${CMAKE_BINARY_DIR}
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
  typeart_add_tidy_target(${target} "${comment}"
    SOURCES ${ARG_SOURCES}
    EXCLUDES ${ARG_EXCLUDES}
    OTHER ${ARG_OTHER} -fix
  )
endfunction()

function(make_tidy_check name sources)
  typeart_add_tidy_target(tidy-run-on-${name}
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
