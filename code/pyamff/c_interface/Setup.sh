#!/bin/bash

# this is the main install script, it differs from the other
# I need to get the pyamff path and set it Makefile.lammps-extra
export PYAMFF_DIR=$(dirname $(cd .. && pwd))
target_DIR=ML-PYAMFF
# sed -i -e "s|^pyanff_SYSPATH =[ \t]*|& ${PYAMFF_DIR}/c_interface/ML-PYAMFF |" Makefile.lammps-extra

cd ${target_DIR}
# create softlinks for lammps
if (test -L "includelink") then
    rm includelink
fi
if (test -L "liblink") then
    rm liblink
fi
if (test -L "Makefile.lammps") then
    rm Makefile.lammps
fi
ln -s ${PYAMFF_DIR}/include includelink
ln -s ${PYAMFF_DIR}/lib liblink
ln -s ${PYAMFF_DIR}/pyamff/c_interface/Makefile.lammps-extra Makefile.lammps
