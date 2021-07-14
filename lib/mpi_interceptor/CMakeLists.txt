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

  add_library(MPIWrapperRuntime SHARED
    ${LIB_SOURCE}
  )
  set_property(TARGET MPIWrapperRuntime PROPERTY OUTPUT_NAME "${PROJECT_NAME}MPIWrapperRuntime")
  add_library(typeart::MPIWrapperRuntime ALIAS MPIWrapperRuntime)


  set_property(TARGET MPIWrapperRuntime PROPERTY C_STANDARD 11)
  set_property(TARGET MPIWrapperRuntime PROPERTY C_STANDARD_REQUIRED TRUE)

  target_define_file_basename(MPIWrapperRuntime)

  target_include_directories(MPIWrapperRuntime
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

  target_include_directories(MPIWrapperRuntime
    SYSTEM
    PRIVATE
      ${LLVM_INCLUDE_DIRS}
  )

  if(ENABLE_TSAN)
    target_tsan_options(MPIWrapperRuntime)
  endif()

  target_link_libraries(MPIWrapperRuntime
    PRIVATE
      typeart::Runtime
      MPI::MPI_C
  )

  install(
    TARGETS MPIWrapperRuntime
    EXPORT ${TARGETS_EXPORT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
else()
  message(STATUS "Cannot build mpi MPIWrapperRuntime lib. MPI found: " ${MPI_C_FOUND} " and python found: " ${PYTHONINTERP_FOUND})
endif()
