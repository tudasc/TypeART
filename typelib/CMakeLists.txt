set(TYPE_LIB_SOURCES
  TypeDB.cpp 
  TypeIO.cpp
)

add_library(musttypelib STATIC 
  ${TYPE_LIB_SOURCES}
)

set_target_properties(musttypelib 
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)
