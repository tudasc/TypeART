function(make_llvm_module name sources)
  # TODO default of include_dirs is private
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