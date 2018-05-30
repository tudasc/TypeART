MPIRUN ?= mpirun
MPICC ?= mpicc
CC ?= cc

all: libtool.so 01_ex 02_ex

libtool.so: tool.c
	$(MPICC) -I ../runtime/ -lmustsupport -L ../build/runtime -shared -fPIC $< -o $@

01_ex: 01_struct_example.c  
	./applyPass.sh $< MustSupportPass.so "-must -must-stats" ../build/lib/ ../build/runtime 1 $@
02_ex: 02_broken_struct_example.c
	./applyPass.sh $< MustSupportPass.so "-must -must-stats" ../build/lib/ ../build/runtime 1 $@

run: all
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../build/runtime $(MPIRUN) -np 1 01_ex
	env LD_PRELOAD=./libtool.so LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../build/runtime $(MPIRUN) -np 1 02_ex

clean:
	rm *~ libtool.so 01_ex 02_ex musttypes