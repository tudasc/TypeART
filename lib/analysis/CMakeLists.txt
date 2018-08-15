set(MEM_PASS_SOURCES
  MemInstFinderPass.cpp
  MemOpVisitor.cpp
  ../support/TypeUtil.cpp
)

make_llvm_module(meminstfinderpass
  "${MEM_PASS_SOURCES}"
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_project_compile_options(meminstfinderpass)
target_project_compile_definitions(meminstfinderpas
  ARG_PRIVATE_DEFS
    LOG_LEVEL=${LOG_LEVEL}
)

if(SHOW_STATS)
  target_project_compile_definitions(typeartpass
    ARG_PRIVATE_DEFS
      LLVM_ENABLE_STATS
  )
endif()

install(
  TARGETS meminstfinderpass
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)