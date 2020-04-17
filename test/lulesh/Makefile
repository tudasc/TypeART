#default build suggestion of MPI + OPENMP with gcc on Livermore machines you might have to change the compiler name

SHELL = /bin/sh
.SUFFIXES: .cc .o .ll

LULESH_EXEC = lulesh2.0

SERCXX = clang++ -DUSE_MPI=0
MPICXX = OMPI_CXX=clang++ mpic++ -DUSE_MPI=1
CXX = $(MPICXX)

TYPEART_RTPATH = ${TYPEART_PATH}/lib
LTYPEART_RT= -L$(TYPEART_PATH)/runtime -ltypeart-rt 

SOURCES2.0 = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc
OBJECTS2.0 = $(SOURCES2.0:.cc=.o)

#Below are reasonable default flags for a serial build w/o OpenMP
CXXFLAGS = -g -I. -Wall
LDFLAGS = -g -O3

PASSES=-load ${TYPEART_PATH}/lib/analysis/meminstfinderpass.so -load ${TYPEART_PATH}/lib/typeartpass.so
CALL_FILTER=-call-filter -call-filter-deep

.cc.o: lulesh.h
ifeq ($(TYPEART), yes)
	$(CXX) ${CXXFLAGS} -O1 -Xclang -disable-llvm-passes -S -emit-llvm $< -o - | opt $(PASSES) -typeart -S | opt -O3 -S | opt $(PASSES) -typeart -typeart-alloca -alloca-array-only=false $(CALL_FILTER) -typeart-no-heap=true | llc -x=ir -filetype=obj -o $@
else
	$(CXX) $(CXXFLAGS) -O3 -c -o $@ $<
endif

all: $(LULESH_EXEC)

lulesh2.0: $(OBJECTS2.0)
	@echo "Linking" 
ifeq ($(TYPEART), yes)
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -lm $(LTYPEART_RT) -o $@
else
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -lm -o $@
endif

clean:
	/bin/rm -f *.o *.s *~ $(OBJECTS) $(LULESH_EXEC) types.yaml
	/bin/rm -rf *.dSYM

tar: clean
	cd .. ; tar cvf lulesh-2.0.tar LULESH-2.0 ; mv lulesh-2.0.tar LULESH-2.0
