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

# mini-cpr variables
MINICPR_PATH = ../../../mini-cpr/src
MINICPR_SOURCES =   $(MINICPR_PATH)/mini-cpr.c      		\
                    $(MINICPR_PATH)/mini-cpr.h      		\
                    $(MINICPR_PATH)/mini-cpr_p.h

# compile and link flags
CXXFLAGS = -g -I. -Wall $(ERROR_DETECTION)
INCLUDES = -I$(MINICPR_PATH) -I$(TYPEART_BASE_DIR)/lib/runtime -I$(TYPEART_BASE_DIR)/lib/typelib
LDFLAGS = -g -O3
CLFLAGS = -Xclang -load -Xclang $(TYPEART_PASS_DIR)/analysis/meminstfinderpass.so -Xclang -load -Xclang $(TYPEART_PASS_DIR)/typeartpass.so -mllvm -alloca-array-only=false -mllvm -typeart-alloca -mllvm -call-filter

# compile commands
SERCXX = clang++ -DUSE_MPI=0
MPICXX = OMPI_CXX=clang++ mpic++ -DUSE_MPI=1
CXX = $(SERCXX)
SERIAL_BUILD_CMD = $(CXX) $(CXXFLAGS) $(INCLUDES) -O1 -Xclang -disable-llvm-passes -S -emit-llvm $< -o - | opt $(PASSES) -typeart -S | opt -O3 -S | opt $(PASSES) -typeart -typeart-alloca -alloca-array-only=false $(CALLFILTER) -typeart-no-heap=true | llc -x=ir -filetype=obj -o $@

all: gol-templ 

extra: gol-papi

gol-templ: gol-templ.o mini-cpr.o
	@echo "Linking"
	$(CXX) gol-templ.o mini-cpr.o $(LDFLAGS) -lm $(LTYPEART_RT)  -o $@

gol-templ.o: main.cpp
	@echo "Building $<"
#	$(CXX) -c $(CXXFLAGS) $(CLFLAGS) -o $@  $<
	$(SERIAL_BUILD_CMD)

gol-papi: main.cpp
	$(CXX) $(CXXFLAGS) -O2 $(PAPI_INC_FLAGS) main.cpp -o gol-papi $(PAPI_LD_FLAGS)

mini-cpr.o: $(MINICPR_SOURCES)
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -O3 -o $@ $<

clean:
	rm -f gol gol-templ gol-papi gol-omp gol-templ-omp gol-papi-finstr mini-cpr.o gol-templ.o types.yaml
	/bin/rm -rf checkpoints/*
