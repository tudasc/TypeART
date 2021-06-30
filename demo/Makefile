# NOTE: env TYPEART_PATH must be set to install base path of TypeART

MPIRUN ?= mpirun
MPICC ?= ${TYPEART_PATH}/bin/typeart-mpicc # instead of standard mpicc

all: libtool.so 01_ex 02_ex toy toy-stack

libtool.so: tool.c
	$(MPICC) -shared -fPIC $< -o $@

01_ex: 01_struct_example.c  
	$(MPICC) -O1 -g -c $< -o $@.o
	$(MPICC) $@.o -o $@

02_ex: 02_broken_struct_example.c
	$(MPICC) -O1 -g -c $< -o $@.o
	$(MPICC) $@.o -o $@

toy-stack: toy.c  
	$(MPICC) -O1 -g -c $< -o $@.o
	$(MPICC) $@.o -o $@

toy: toy.c
	$(MPICC) -DNOSTACK -O1 -g -c $< -o $@.o
	$(MPICC) $@.o -o $@

runtoy: toy toy-stack
	./toy
	./toy-stack

run: run01 run02

run01: 01_ex libtool.so
	env LD_PRELOAD=./libtool.so $(MPIRUN) -np 1 01_ex
run02: 02_ex libtool.so
	env LD_PRELOAD=./libtool.so $(MPIRUN) -np 1 02_ex

clean:
	rm *.o libtool.so 01_ex 02_ex toy toy-stack types.yaml
