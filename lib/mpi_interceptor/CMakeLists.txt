find_package(MPI QUIET)
find_package(PythonInterp QUIET)

if(${MPI_C_FOUND} AND ${PYTHONINTERP_FOUND})
  set(LIB_SOURCE
    ${CMAKE_CURRENT_BINARY_DIR}/mpi_interceptor_rt.c
  )
  
  set(LIB_TMPL
    mpi_interceptor_tmpl.impl
  )

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} wrap.py ${LIB_TMPL} -o ${LIB_SOURCE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_library(interceptor-rt SHARED
    ${LIB_SOURCE}
  )

  target_define_file_basename(interceptor-rt)
  
  target_include_directories(interceptor-rt
    PUBLIC
      ${PROJECT_SOURCE_DIR}/lib
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  target_include_directories(interceptor-rt
    SYSTEM 
    PRIVATE
      ${MPI_C_INCLUDE_PATH}
      ${LLVM_INCLUDE_DIRS}
  )

  if(ENABLE_TSAN)
    target_compile_options(interceptor-rt PRIVATE -fsanitize=thread)
    target_link_options(interceptor-rt PRIVATE -fsanitize=thread)
  endif()

  target_link_libraries(interceptor-rt
    typeart-rt
    ${MPI_C_LIBRARIES}
  )

  install(
    TARGETS interceptor-rt
    EXPORT ${TARGETS_EXPORT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
  )
else()
  message(STATUS "Cannot build mpi interceptor-rt lib. MPI found: " ${MPI_C_FOUND} " and python found: " ${PYTHONINTERP_FOUND})
endif()
