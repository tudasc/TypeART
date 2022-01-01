function(make_llvm_module name sources)
  # TODO default of include_dirs is private
  cmake_parse_arguments(ARG "" "" "INCLUDE_DIRS;DEPENDS;LINK_LIBS" ${ARGN})

  # TODO find out which LLVM versions support the respective calls
  if(${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL "10.0.0")
    add_llvm_pass_plugin(${name}
      ${sources}
      LINK_LIBS LLVMDemangle ${ARG_LINK_LIBS}
      DEPENDS ${ARG_DEPENDS}
    )
    target_compile_definitions(${name}
      PRIVATE
        LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
    )
  elseif(${LLVM_PACKAGE_VERSION} VERSION_EQUAL "6.0")
    add_llvm_loadable_module(${name}
      ${sources}
      LINK_LIBS LLVMDemangle ${ARG_LINK_LIBS}
      DEPENDS ${ARG_DEPENDS}
    )
    target_compile_definitions(${name}
      PRIVATE
        LLVM_VERSION=6
    )
  endif()

  target_include_directories(${name}
    SYSTEM
    PRIVATE
    ${LLVM_INCLUDE_DIRS}
  )

  if(ARG_INCLUDE_DIRS)
    target_include_directories(${name}
      PRIVATE
      ${ARG_INCLUDE_DIRS}
    )
  endif()

  target_define_file_basename(${name})

  target_compile_definitions(${name}
    PRIVATE
    ${LLVM_DEFINITIONS}
  )

  make_tidy_check(${name}
    ${sources}
  )
endfunction()

function(typeart_find_llvm_progs target names default)
  find_program(
    target-prog
    NAMES ${names}
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH
  )

  if(NOT target-prog)
    set(${target}
        ${default}
        PARENT_SCOPE
    )
    message(
      STATUS
        "Did not find clang program ${names} in ${LLVM_TOOLS_BINARY_DIR}. Using def. value: ${default}"
    )
  else()
    set(${target}
        ${target-prog}
        PARENT_SCOPE
    )
  endif()
  unset(target-prog CACHE)
endfunction()
