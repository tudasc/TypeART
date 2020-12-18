function(configure_typeart_script input output)
  set(TYPEART_PROJECT_DIR ${PROJECT_SOURCE_DIR})
  set(TYPEART_BINARY_DIR ${CMAKE_BINARY_DIR})
  set(TYPEART_SCRIPT_DIR ${PROJECT_SOURCE_DIR}/scripts)
  set(TYPEART_PASS_DIR ${CMAKE_BINARY_DIR}/lib/passes)
  set(TYPEART_ANALYSIS_PASS_DIR ${TYPEART_PASS_DIR}/analysis)
  set(TYPEART_RT_DIR ${CMAKE_BINARY_DIR}/lib/runtime)
  set(TYPEART_ANALYSIS_PLUGIN meminstfinderpass.so)
  set(TYPEART_PLUGIN typeartpass.so)
  set(TYPEART_INCLUDE_DIRS -I${PROJECT_SOURCE_DIR}/lib/typelib)

  # To get execute permission: create run.sh in the binary dir, and copy it to scripts folder with permission
  configure_file(${input} ${CMAKE_CURRENT_BINARY_DIR}/${output} @ONLY)
  file(
    COPY ${CMAKE_CURRENT_BINARY_DIR}/${output}
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
endfunction()

configure_typeart_script(run.sh.in run.sh)
configure_typeart_script(apply.sh.in apply.sh)


#configure_file(clang-typeart.in clang-typeart)
#configure_file(clang-typeart++.in clang-typeart++)

#install(
#  PROGRAMS
#    ${CMAKE_CURRENT_BINARY_DIR}/clang-typeart
#    ${CMAKE_CURRENT_BINARY_DIR}/clang-typeart++
#  DESTINATION bin
#)
