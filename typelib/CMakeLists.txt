set(TYPE_LIB_SOURCES
  TypeDB.cpp 
  TypeIO.cpp
)

add_library(typelib STATIC
  ${TYPE_LIB_SOURCES}
)

#target_link_libraries(musttypelib
#  LLVMCore
#  LLVMSupport
#)

set_target_properties(typelib
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(typelib
  SYSTEM 
  PRIVATE 
    ${LLVM_INCLUDE_DIRS}
)

make_tidy_check(typelib
  "${TYPE_LIB_SOURCES}"
)
  
install(
  TARGETS typelib
  EXPORT typeart_typelib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION
    include 
)

install(
  FILES TypeInterface.h
  DESTINATION include
)

install(
  EXPORT typeart_typelib
  NAMESPACE typeart_
  DESTINATION lib/typeart
)