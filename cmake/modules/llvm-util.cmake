function(make_llvm_module name sources)
  cmake_parse_arguments(ARG "" "" "INCLUDE_DIRS;DEPENDS;LINK_LIBS" ${ARGN})
  
  add_llvm_loadable_module(${name}
    ${sources}
    LINK_LIBS LLVMDemangle ${ARG_LINK_LIBS}
    DEPENDS ${ARG_DEPENDS}
  )
  
  target_include_directories(${name}
    SYSTEM 
    PUBLIC 
    ${LLVM_INCLUDE_DIRS}
  )
  
  if(ARG_INCLUDE_DIRS)
    target_include_directories(${name}
      PUBLIC 
      ${ARG_INCLUDE_DIRS}
    )
  endif()
  
  target_define_file_basename(${name})
  
  target_compile_definitions(${name}
    PRIVATE
    ${LLVM_DEFINITIONS}
  )

  add_tidy_target(tidy-run-on-${name}
    "Clang-tidy run on ${name} translation units"
    SOURCES ${sources}
    OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_tidy_fix_target(tidy-fix-on-${name}
    "Clang-tidy run with fixes on ${name} translation units"
    SOURCES ${sources}
    OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR} -checks=-*,modernize-*,llvm-namespace-comment,google-explicit-constructor
  )
endfunction()