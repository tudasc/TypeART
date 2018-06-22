MPIRUN ?= mpirun
MPICC ?= OMPI_CC=clang mpicc
CC ?= cc
TYPEART_PASS_DIR = ../build/lib
TYPEART_RUNTIME_DIR = ../build/runtime
CLFLAGS = -ltypeart -L$(TYPEART_RUNTIME_DIR)
CFLAGS = -Xclang -load -Xclang $(TYPEART_PASS_DIR)/analysis/MemInstFinderPass.so -Xclang -load -Xclang $(TYPEART_PASS_DIR)/TypeArtPass.so


all: libtool.so 01_ex 02_ex

libtool.so: tool.c
	$(MPICC) -I../runtime/ -I../typelib/  $(CLFLAGS) -shared -fPIC $< -o $@

01_ex: 01_struct_example.c  
	$(MPICC) $(CFLAGS) $(CLFLAGS) $< -o $@ -g
02_ex: 02_broken_struct_example.c
	$(MPICC) $(CFLAGS) $(CLFLAGS) $< -o $@ -g

run: all
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) $(MPIRUN) -np 1 01_ex
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(TYPEART_RUNTIME_DIR) $(MPIRUN) -np 1 02_ex

clean:
	rm *~ libtool.so 01_ex 02_ex types.yaml
