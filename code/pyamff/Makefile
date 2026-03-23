#!/bin/make -f
ROOT_DIR=..
INC_DIR=${ROOT_DIR}/include
LIB_DIR=${ROOT_DIR}/lib
OBJECTS =  \
        fortran/fptype.o \
        fortran/atoms.o \
        fortran/neighborlist.o \
        fortran/behlerParrinello.o \
        fortran/nntype.o \
        fortran/trainType.o \
        fortran/normalization.o \
        fortran/fingerprints.o \
        fortran/neuralnetwork.o \
        fortran/nnmodule.o \
        fortran/pyamffCalc.o

LIB := libAMFF.a
FC=gfortran
AR=ar cru
RANLIB=ranlib


all: main libAMFF.a

main:
	f2py -c --opt="-O2" --f90flags="-fmax-stack-var-size=50000" -m fmodules \
	    fortran/fptype.f90 \
		fortran/atoms.f90 \
		fortran/neighborlist.f90 \
		fortran/behlerParrinello.f90 \
		fortran/nntype.f90 \
		fortran/trainType.f90 \
		fortran/normalization.f90\
		fortran/fingerprints.f90 \
		fortran/neuralnetwork.f90 \
		fortran/nnmodule.f90 \
		fortran/pyamffCalc.f90 \

libAMFF.a:
#TARGET = libAMFF.a
ifdef EON
    OBJECTS =  \
        fortran/fptype.o \
        fortran/atoms.o \
        fortran/neighborlist.o \
        fortran/behlerParrinello.o \
        fortran/nntype.o \
        fortran/trainType.o \
        fortran/normalization.o \
        fortran/fingerprints.o \
        fortran/neuralnetwork.o \
        fortran/nnmodule.o \
        fortran/pyamffCalc.o

    FC=gfortran
    AR=ar cru
    RANLIB=ranlib
    %.o: %.f90
		$(FC) -c -O2 $< -o $@

    LIB := libAMFF.a

    $(LIB) : $(OBJECTS)
		$(AR) $@ $^
		$(RANLIB) $@
	echo "done eon"
endif

clean:
	rm -f fmodules.*so
	rm -f fortran/*.o
	rm -f *.mod
	rm -f libAMFF.a
	rm -f ${LIB_DIR}/*.a
	cd c_interface && make clean


lammps: main library
	@cd c_interface && make && $(SHELL) Setup.sh
ifdef LAMMPS_DIR
	@echo "LAMMPS_DIR=${LAMMPS_DIR}"
	@cd c_interface && cp -r ML-PYAMFF ${LAMMPS_DIR}/src
	@mkdir -p ${LAMMPS_DIR}/lib/pyamff
	@cd ${LAMMPS_DIR} && find src/ML-PYAMFF -type l -exec mv {} lib/pyamff/ \;
	@echo "Library and Header links are in ${LAMMPS_DIR}/lib/pyamff"
else
	@echo "LAMMPS_DIR undefined. You will need to copy ML-PYAMFF to src manually."
	@echo "Please refer to README"
endif
	@echo "lammps interface building is done"

library : $(OBJECTS)
	$(AR) $(LIB) $^
	$(RANLIB) $(LIB)
	cp $(LIB) ${LIB_DIR}

%.o: %.f90
	$(FC) -c -O2 $< -o $@
