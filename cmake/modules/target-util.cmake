function(typeart_target_compile_options target)
  cmake_parse_arguments(ARG "" "" "PRIVATE_FLAGS;PUBLIC_FLAGS" ${ARGN})

  if (TYPEART_IS_TOP_LEVEL)
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -pedantic
      -Wunreachable-code -Wwrite-strings
      -Wpointer-arith -Wcast-align
      -Wcast-qual -Wno-unused-parameter
      -Wunused -Wshadow
      -Wformat=2 -Wundef -Werror=float-equal
    )
  endif()

  target_compile_definitions(${target} PRIVATE "LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}")

  if (ARG_PRIVATE_FLAGS)
    target_compile_options(${target} PRIVATE
      "${ARG_PRIVATE_FLAGS}"
    )
  endif ()

  if (ARG_PUBLIC_FLAGS)
    target_compile_options(${target} PUBLIC
      "${ARG_PUBLIC_FLAGS}"
    )
  endif ()
endfunction()

function(typeart_target_compile_definitions target)
  cmake_parse_arguments(ARG "" "" "PRIVATE_DEFS;PUBLIC_DEFS" ${ARGN})

  if (ARG_PRIVATE_DEFS)
    target_compile_definitions(${target} PRIVATE
      "${ARG_PRIVATE_DEFS}"
    )
  endif ()

  if (ARG_PUBLIC_DEFS)
    target_compile_definitions(${target} PUBLIC
      "${ARG_PUBLIC_DEFS}"
    )
  endif ()
endfunction()

function (typeart_target_generate_file input output)
  file(READ ${input} contents)
  string(CONFIGURE "${contents}" contents @ONLY)
  file(GENERATE
    OUTPUT
      ${output}
    CONTENT
      "${contents}"
    FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ
      WORLD_READ
  )
endfunction()
