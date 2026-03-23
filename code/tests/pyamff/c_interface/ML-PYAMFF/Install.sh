# Installation script required for LAMMPS Packages
# mode = uninstall (0),install(1), and update (3)

mode=$1

#files update; may need to be tweaked
for file in *.cpp *.h; do
    if (test -f ${file}) then
        if (test $mode = 0) then
            rm -f ../${file}
        elif (! cmp -s ${file} ../${file}) then
            if (! test -f ../${file} || test ${mode} = 2 ) then
                cp ${file} ..
                if (test ${mode} = 2) then
                    echo "updating src/${file}"
                fi
            fi
        fi
    fi
done

# edit Makefiles to add or remove package info
if (test $mode = 1) then
    if (test -e ../Makefile.package) then
        sed -i -e 's/[^ \t]*pyamff[^ \t]* //g' ../Makefile.package # remove all instances of pyamff
        sed -i -e 's|^PKG_INC =[ \t]*|&-I../../lib/pyamff/includelink |' ../Makefile.package # update include links
        sed -i -e 's|^PKG_PATH =[ \t]*|&-L../../lib/pyamff/liblink |' ../Makefile.package #update lib links
        sed -i -e 's|^PKG_SYSINC =[ \t]*|&$(pyamff_SYSINC) |' ../Makefile.package # add include directory
        sed -i -e 's|^PKG_SYSLIB =[ \t]*|&$(pyamff_SYSLIB) |' ../Makefile.package # add lib directory
        sed -i -e 's|^PKG_SYSPATH =[ \t]*|&$(pyamff_SYSPATH) |' ../Makefile.package # add path
        sed -i -e 's|^PACKAGE =[ \t]*|&ml-pyamff |' ../Makefile # add ml-pyamff to packages list
    fi

    if (test -e ../Makefile.package.settings) then
        sed -i -e '/^include.*pyamff.*$/d' ../Makefile.package.settings
        sed -i -e '4 i \
        include ..\/..\/lib\/pyamff\/Makefile.lammps
        ' ../Makefile.package.settings
    fi
elif (test $mode = 0) then
    if (test -e ../Makefile.package) then
        sed -i -e 's/[^ \t]*pyamff[^ \t]* //g' ../Makefile.package # remove all instances of pyamff
    fi
    if (test -e ../Makefile.package.settings) then # this doesn't remove
        sed -i -e '/^[ \t]*include.*pyamff.*$/d' ../Makefile.package.settings
    fi
fi
