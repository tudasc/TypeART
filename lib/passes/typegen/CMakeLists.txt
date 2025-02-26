if(${LLVM_VERSION_MAJOR} VERSION_LESS_EQUAL "14")
  add_subdirectory(ir)
endif()
add_subdirectory(dimeta)