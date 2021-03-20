


MPICXX=mpicxx
VELOC_INSTALL=${VELOC_PATH}
FTI_INSTALL=${FTI_PATH}
TYPEART_BASE=../../


ifneq ($(VELOC_INSTALL),)
  INCLUDE_FLAGS=-I$(VELOC_INSTALL)/include -I$(TYPEART_BASE)/lib/typelib -I$(TYPEART_BASE)/lib/runtime/tycart -I$(TYPEART_BASE)/lib/runtime -I$(TYPEART_BASE)/lib/passes
  CPR_FLAGS=-DWITH_VELOC
  CPR_LIBS=-L$(VELOC_INSTALL)/lib -lveloc-client -lveloc-modules
else ifneq ($(FTI_INSTALL),)
  INCLUDE_FLAGS=-I$(FTI_INSTALL)/include -I$(TYPEART_BASE)/lib/typelib -I$(TYPEART_BASE)/lib/runtime/tycart -I$(TYPEART_BASE)/lib/runtime -I$(TYPEART_BASE)/lib/passes
  CPR_FLAGS=-DWITH_FTI
  CPR_LIBS=-L$(FTI_INSTALL)/lib -lfti
else
	$(error Need to specify either VELOC_PATH or FTI_PATH)
endif


TYPEART_LIBS=-L$(TYPEART_BASE)/build/lib/runtime/tycart -ltycart-rt  -L$(TYPEART_BASE)/build/lib/runtime -ltypeart-rt


all: 01_noerr 02_noerr 10_err 11_err 12_err

info:
	echo "call with 'make run' and adjust VeloC path using 'make run VELOC_INSTALL=/path'"

01_noerr: 01_noerr.cpp
	$(MPICXX) $(CPR_FLAGS) $(INCLUDE_FLAGS) $< $(TYPEART_LIBS) $(CPR_LIBS) -o $@

02_noerr: 02_noerr.cpp
	$(MPICXX) $(CPR_FLAGS)  $(INCLUDE_FLAGS) $< $(TYPEART_LIBS) $(CPR_LIBS) -o $@

10_err: 10_err.cpp
	$(MPICXX) $(CPR_FLAGS)  $(INCLUDE_FLAGS) $< $(TYPEART_LIBS) $(CPR_LIBS) -o $@

11_err: 11_err.cpp
	$(MPICXX)  $(CPR_FLAGS) $(INCLUDE_FLAGS) $< $(TYPEART_LIBS) $(CPR_LIBS) -o $@

12_err: 12_err.cpp
	$(MPICXX)  $(CPR_FLAGS) $(INCLUDE_FLAGS) $< $(TYPEART_LIBS) $(CPR_LIBS) -o $@

run: all
	./run_all.sh


clean:
	rm -f 01_noerr 02_noerr 10_err 11_err log.err
