function(typeart_target_sanitizer_options target)
  cmake_parse_arguments(ARG "" "" "SAN_FLAGS;LINK_FLAGS" ${ARGN})

  target_compile_options(${target}
    PRIVATE
      ${ARG_SAN_FLAGS}
  )

  target_link_options(${target}
    PUBLIC
    ${ARG_SAN_FLAGS}
  )

  if (ARG_LINK_FLAGS)
    target_link_options(${target}
      PUBLIC
      "${ARG_LINK_FLAGS}"
    )
  endif()

endfunction()

function(typeart_target_tsan_flags flags)
  set(${flags} -fsanitize=thread PARENT_SCOPE)
endfunction()

function(typeart_target_asan_flags flags)
  set(${flags} -fsanitize=address -fno-omit-frame-pointer PARENT_SCOPE)
endfunction()

function(typeart_target_ubsan_flags flags)
  set(${flags} -fsanitize=undefined -fno-sanitize=vptr,function -fno-sanitize-recover=undefined,integer PARENT_SCOPE)
endfunction()

function(typeart_target_asan_options target)
  typeart_target_asan_flags(asan_f)
  typeart_target_sanitizer_options(${target}
    SAN_FLAGS
      ${asan_f}
  )
endfunction()

function(typeart_target_ubsan_options target)
  typeart_target_ubsan_flags(ubsan_f)
  typeart_target_sanitizer_options(${target}
    SAN_FLAGS
      ${ubsan_f}
  )
endfunction()

function(typeart_target_tsan_options target)
  typeart_target_tsan_flags(tsan_f)
  typeart_target_sanitizer_options(${target}
    SAN_FLAGS
      "${tsan_f}"
  )
endfunction()
