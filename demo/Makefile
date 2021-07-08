# NOTE: PATH needs to include bin dir of typeart-mpicc wrapper

MPIRUN ?= mpirun
MPICC ?= typeart-mpicc # instead of standard mpicc

all: libtool.so 01_ex 02_ex toy toy-stack

libtool.so: tool.c
	$(MPICC) -shared -fPIC $< -o $@

demo: 01_struct_example.c
	$(MPICC) -O1 -g -c $< -o $@.o
	$(MPICC) $@.o -o $@

demo_broken: 02_broken_struct_example.c
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

run: run-demo run-demo_broken

run-demo: demo libtool.so
	env LD_PRELOAD=./libtool.so $(MPIRUN) -np 1 demo
run-demo_broken: demo_broken libtool.so
	env LD_PRELOAD=./libtool.so $(MPIRUN) -np 1 demo_broken

clean:
	rm -f *.o libtool.so 01_ex 02_ex toy toy-stack types.yaml
