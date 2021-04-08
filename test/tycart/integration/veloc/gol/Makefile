# build tycart integration testing app
# GameOfLife code base by jplehr: https://github.com/jplehr/GameOfLife

# variables for error detection script
ACTUAL = $(ACTUAL_TYPE) $(ACTUAL_X) $(ACTUAL_Y)
EXPECTED = $(EXPECTED_TYPE) $(EXPECTED_SIZE) $(EXPECTED_X) $(EXPECTED_Y)
ERROR_DETECTION = $(ACTUAL) $(EXPECTED)

# typeart
TYPEART_BASE_DIR = ../../../../..
TYPEART_RTPATH = ../../../../../build/lib/runtime
TYPEART_PASS_DIR = ../../../../../build/lib/passes
LTYPEART_RT= -L$(TYPEART_RTPATH) -ltypeart-rt
PASSES = -load $(TYPEART_PASS_DIR)/analysis/meminstfinderpass.so -load $(TYPEART_PASS_DIR)/typeartpass.so -typeart-stats
CALLFILTER = -call-filter -call-filter-deep

TYPECHECK_MACRO = ../../../../../lib/runtime/tycart/typecheck_macro.h

# veloc variables: VELOC_PATH needs to be set in shell variable
VELOC_LIB = -L$(VELOC_PATH)/lib -lveloc-client -lveloc-modules -L$(VELOC_PATH)/lib64 -ler -lrankstr -laxl -lkvtree -lredset -lshuffile

# compile and link flags
CXXFLAGS = -g -I. -Wall $(ERROR_DETECTION)
INCLUDES = -I$(VELOC_PATH)/include -I$(TYPEART_BASE_DIR)/lib/runtime -I$(TYPEART_BASE_DIR)/lib/typelib
LDFLAGS = -g -O3
CLFLAGS = -Xclang -load -Xclang $(TYPEART_PASS_DIR)/analysis/meminstfinderpass.so -Xclang -load -Xclang $(TYPEART_PASS_DIR)/typeartpass.so -mllvm -alloca-array-only=false -mllvm -typeart-alloca -mllvm -call-filter

# compile commands
SERCXX = clang++ -DUSE_MPI=0
MPICXX = OMPI_CXX=clang++ mpic++ -DUSE_MPI=1
CXX = $(MPICXX)
MPI_BUILD_CMD = $(MPICXX) $(CXXFLAGS) $(INCLUDES) -O1 -Xclang -disable-llvm-passes -S -emit-llvm $< -o - | opt $(PASSES) -typeart -S | opt -O3 -S | opt $(PASSES) -typeart -typeart-alloca -alloca-array-only=false $(CALLFILTER) -typeart-no-heap=true | llc -x=ir -filetype=obj -o $@

all: gol-templ 

extra: gol-papi

gol-templ: gol-templ.o veloc.config
	@echo "Linking"
	$(CXX) gol-templ.o $(LDFLAGS) -lm $(VELOC_LIB) -lm $(LTYPEART_RT) -o $@

gol-templ.o: main.cpp
	@echo "Building $<"
#	$(CXX) -c $(CXXFLAGS) $(CLFLAGS) -o $@  $<
	$(MPI_BUILD_CMD)

veloc.config: veloc.config.original
	cp veloc.config.original veloc.config

clean:
	rm -f gol gol-templ gol-papi gol-omp gol-templ-omp gol-papi-finstr gol-templ.o veloc.config types.yaml
	/bin/rm -rf persistent/* scratch/*
