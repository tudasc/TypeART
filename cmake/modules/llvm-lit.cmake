find_package(PythonInterp QUIET)

if(LLVM_EXTERNAL_LIT)
  message(STATUS "External lit path is used")
  get_llvm_lit_path(
      lit_base_dir
      lit_file_name
      ALLOW_EXTERNAL
  )
  set(path_to_lit ${lit_file_name})
  if(lit_base_dir)
    set(path_to_lit ${lit_base_dir}/${path_to_lit})
  endif()
  set(LIT_COMMAND_I "${PYTHON_EXECUTABLE};${path_to_lit}")
endif()

if(NOT LIT_COMMAND_I)
  find_program(LLVM_LIT_PATH
      NAME llvm-lit
      HINTS ${LLVM_TOOLS_BINARY_DIR}
      PATHS ${LLVM_ROOT_DIR}/bin /usr/bin /usr/local/bin /opt/local/bin /usr/lib
      )

  if(LLVM_LIT_PATH)
    get_filename_component(path_to_llvm_lit ${LLVM_LIT_PATH} ABSOLUTE CACHE)
    set(LIT_COMMAND_I ${path_to_llvm_lit})
    set(LLVM_EXTERNAL_LIT ${LLVM_LIT_PATH})
  else()
    message(WARNING "No llvm lit is available")
  endif()
endif()

message(STATUS "llvm lit command is set to ${LIT_COMMAND_I}")