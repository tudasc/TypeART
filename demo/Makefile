BUILDDIR ?= ../build
MPIRUN ?= mpirun
MPICC ?= OMPI_CC=clang mpicc
CC ?= cc
TYPEART_PASS_DIR = $(BUILDDIR)/lib
TYPEART_RUNTIME_DIR = $(BUILDDIR)/runtime
CLFLAGS = -L$(TYPEART_RUNTIME_DIR) -ltypeart-rt
CFLAGS = -Xclang -load -Xclang $(TYPEART_PASS_DIR)/analysis/meminstfinderpass.so -Xclang -load -Xclang $(TYPEART_PASS_DIR)/typeartpass.so


all: libtool.so 01_ex 02_ex 03_ex toy toy-stack toy-nostack

libtool.so: tool.c
	$(MPICC) -I../runtime/ -I../typelib/  $(CLFLAGS) -shared -fPIC $< -o $@

01_ex: 01_struct_example.c  
	$(MPICC) $(CFLAGS) $(CLFLAGS) $< -o $@ -g
02_ex: 02_broken_struct_example.c
	$(MPICC) $(CFLAGS) $(CLFLAGS) $< -o $@ -g
03_ex: 03_array_example.c
	$(MPICC) $(CFLAGS) $(CLFLAGS) $< -o $@ -g

toy: toy.c  
	$(MPICC) -I../runtime/ -I../typelib/  $(CFLAGS) $(CLFLAGS) $< -o $@ -g

toy-stack: toy.c  
	$(MPICC) -I../runtime/ -I../typelib/ -S -emit-llvm $< -o $@.ll -g
	opt -load $(TYPEART_PASS_DIR)/analysis/meminstfinderpass.so -load $(TYPEART_PASS_DIR)/typeartpass.so -typeart -typeart-alloca < $@.ll -o $@.ll > /dev/null
	llc $@.ll -o $@.s
	$(MPICC) $(CLFLAGS) $@.s -o $@  

toy-nostack: toy.c  
	$(MPICC) -I../runtime/ -I../typelib/ -DNOSTACK -S -emit-llvm $< -o $@.ll -g
	opt -load $(TYPEART_PASS_DIR)/analysis/meminstfinderpass.so -load $(TYPEART_PASS_DIR)/typeartpass.so -typeart < $@.ll -o $@.ll > /dev/null
	llc $@.ll -o $@.s
	$(MPICC) $(CLFLAGS) $@.s -o $@  


runtoy: toy toy-stack toy-nostack
	env LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) ./toy
	env LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) ./toy-stack
	env LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) ./toy-nostack

run: run01 run02 run03

run01: 01_ex libtool.so
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) $(MPIRUN) -np 1 01_ex
run02: 02_ex libtool.so
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) $(MPIRUN) -np 1 02_ex
run03: 03_ex libtool.so
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) $(MPIRUN) -np 4 03_ex


clean:
	rm *~ libtool.so 01_ex 02_ex types.yaml
