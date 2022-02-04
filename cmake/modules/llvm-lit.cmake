find_package(Python3 QUIET)

if(LLVM_EXTERNAL_LIT)
  message(STATUS "External lit path is used")
  get_llvm_lit_path(
    lit_base_dir
    lit_file_name
    ALLOW_EXTERNAL
  )
  set(PATH_TO_LIT ${lit_file_name})
  if(lit_base_dir)
    set(PATH_TO_LIT ${lit_base_dir}/${PATH_TO_LIT})
  endif()
  set(LIT_COMMAND_I "${Python3_EXECUTABLE};${PATH_TO_LIT}")
endif()

if(NOT LIT_COMMAND_I)
  find_program(LLVM_LIT_PATH
    NAME llvm-lit lit lit.py
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    PATHS ${LLVM_ROOT_DIR}/bin /usr/bin /usr/local/bin /opt/local/bin /usr/lib
  )

  if(LLVM_LIT_PATH)
    get_filename_component(TYPEART_LLVM_LIT_PATH ${LLVM_LIT_PATH} ABSOLUTE CACHE)
    set(LIT_COMMAND_I ${TYPEART_LLVM_LIT_PATH})
    set(LLVM_EXTERNAL_LIT ${LLVM_LIT_PATH})
  else()
    message(WARNING "No llvm lit is available")
  endif()
endif()

mark_as_advanced(TYPEART_LLVM_LIT_PATH)

message(STATUS "llvm lit command is set to ${LIT_COMMAND_I}")
