set(TYPE_LIB_SOURCES
  TypeDB.cpp 
  TypeIO.cpp
)

add_library(musttypelib STATIC 
  ${TYPE_LIB_SOURCES}
)

#target_link_libraries(musttypelib
#  LLVMCore
#  LLVMSupport
#)

set_target_properties(musttypelib 
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

target_include_directories(musttypelib
  SYSTEM 
  PRIVATE 
  ${LLVM_INCLUDE_DIRS}
)

make_tidy_check(musttypelib
  "${TYPE_LIB_SOURCES}"
)
  
