add_subdirectory(analysis)

set(PASS_SOURCES
  TypeARTPass.cpp
  support/TypeUtil.cpp
  TypeManager.cpp
)

make_llvm_module(typeartpass
  "${PASS_SOURCES}"
  LINK_LIBS
    typelib
  DEPENDS
    meminstfinderpass
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/typelib
)

set(PROJECT_NAME typeart_llvm)
set(TARGETS_EXPORT_NAME typeart_llvm)

install(
  TARGETS typeartpass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# also handles subdir meminstfinder
install(
  EXPORT ${TARGETS_EXPORT_NAME}
  NAMESPACE typeart::
  DESTINATION lib/cmake
)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/typeart_llvmConfig.cmake
  INSTALL_DESTINATION lib/cmake
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/typeart_llvmConfig.cmake
  DESTINATION lib/cmake
)
