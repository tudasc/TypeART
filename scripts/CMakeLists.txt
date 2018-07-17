configure_file(clang-typeart.in clang-typeart)
configure_file(clang-typeart++.in clang-typeart++)

install(
  PROGRAMS 
    ${CMAKE_CURRENT_BINARY_DIR}/clang-typeart
    ${CMAKE_CURRENT_BINARY_DIR}/clang-typeart++
  DESTINATION bin
)
