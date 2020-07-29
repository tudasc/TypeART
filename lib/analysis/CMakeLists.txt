set(MEM_PASS_SOURCES
  MemInstFinderPass.cpp
  MemOpVisitor.cpp
  ../support/TypeUtil.cpp
  ../support/ModuleDataManager.cpp
  ../support/ModuleManagerBase.cpp
  ../support/CGInterface.cpp
)

make_llvm_module(meminstfinderpass
  "${MEM_PASS_SOURCES}"
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/datalib
    ${PROJECT_SOURCE_DIR}/typelib
)

target_project_compile_options(meminstfinderpass)
target_project_compile_definitions(meminstfinderpass
  PRIVATE_DEFS
    LOG_LEVEL=${LOG_LEVEL}
)

if(SHOW_STATS)
  target_project_compile_definitions(meminstfinderpass
    PRIVATE_DEFS 
      LLVM_ENABLE_STATS
  )
endif()

install(
  TARGETS meminstfinderpass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)