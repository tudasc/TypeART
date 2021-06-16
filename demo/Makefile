# NOTE: env TYPEART_PATH must be set to install base path of TypeART

MPIRUN ?= mpirun
MPICC ?= OMPI_CC=clang mpicc
CC ?= cc

TYPEART_LIBPATH = ${TYPEART_PATH}/lib
TYPEART_INCPATH = ${TYPEART_PATH}/include

typeart_plugin = -load $(TYPEART_LIBPATH)/meminstfinderpass.so -load $(TYPEART_LIBPATH)/typeartpass.so -typeart
typeart_stack_mode_args = -typeart-no-heap -typeart-alloca -typeart-stats -S
typeart_heap_mode_args = -typeart-no-heap=false -typeart-stats -S

CLFLAGS = -L$(TYPEART_LIBPATH) -ltypeart-rt

all: libtool.so 01_ex 02_ex toy toy-stack

libtool.so: tool.c
	$(MPICC) -I$(TYPEART_INCPATH) $(CLFLAGS) -shared -fPIC $< -o $@

01_ex: 01_struct_example.c  
	$(MPICC) -I$(TYPEART_INCPATH) -g -O1 -Xclang -disable-llvm-passes -S -emit-llvm $< -o - | opt $(typeart_plugin) $(typeart_heap_mode_args) | opt -O1 -S | opt $(typeart_plugin) $(typeart_stack_mode_args) | llc -x=ir -filetype=obj -o $@.o
	$(MPICC) $(CLFLAGS) $@.o -o $@
02_ex: 02_broken_struct_example.c
	$(MPICC) -I$(TYPEART_INCPATH) -g -O1 -Xclang -disable-llvm-passes -S -emit-llvm $< -o - | opt $(typeart_plugin) $(typeart_heap_mode_args) | opt -O1 -S | opt $(typeart_plugin) $(typeart_stack_mode_args) | llc -x=ir -filetype=obj -o $@.o
	$(MPICC) $(CLFLAGS) $@.o -o $@

toy-stack: toy.c  
	$(MPICC) -I$(TYPEART_INCPATH) -g -O1 -Xclang -disable-llvm-passes -S -emit-llvm $< -o - | opt $(typeart_plugin) $(typeart_heap_mode_args) | opt -O1 -S | opt $(typeart_plugin) $(typeart_stack_mode_args) | llc -x=ir -filetype=obj -o $@.o
	$(MPICC) $(CLFLAGS) $@.o -o $@

toy: toy.c
	$(MPICC) -I$(TYPEART_INCPATH) -DNOSTACK -g -O1 -Xclang -disable-llvm-passes -S -emit-llvm $< -o - | opt $(typeart_plugin) $(typeart_heap_mode_args) | opt -O1 -S | opt $(typeart_plugin) $(typeart_stack_mode_args) | llc -x=ir -filetype=obj -o $@.o
	$(MPICC) $(CLFLAGS) $@.o -o $@

runtoy: toy toy-stack
	env LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_LIBPATH) ./toy
	env LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_LIBPATH) ./toy-stack

run: run01 run02

run01: 01_ex libtool.so
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_LIBPATH) $(MPIRUN) -np 1 01_ex
run02: 02_ex libtool.so
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_LIBPATH) $(MPIRUN) -np 1 02_ex

clean:
	rm *.o libtool.so 01_ex 02_ex toy toy-stack types.yaml
