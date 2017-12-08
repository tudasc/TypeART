
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)


add_llvm_loadable_module(
	MUSTSupportPass.cpp
	TypeHelper.cpp
	)

