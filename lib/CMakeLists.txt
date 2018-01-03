list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)

set(PASS_SOURCES
 	MUSTSupportPass.cpp
	TypeUtil.cpp
	MemOpVisitor.cpp
)

add_llvm_loadable_module(MustSupportPass
	${PASS_SOURCES}
)



add_tidy_target(tidy-run-on-pass
  "Clang-tidy run on MUST support pass translation units"
  SOURCES ${PASS_SOURCES}
  OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR}
)

add_tidy_fix_target(tidy-fix-on-pass
  "Clang-tidy run with fixes on  MUST support pass translation units"
  SOURCES ${PASS_SOURCES}
  OTHER --header-filter=${CMAKE_CURRENT_SOURCE_DIR} -checks=-*,modernize-*,llvm-namespace-comment,google-explicit-constructor
)