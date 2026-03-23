**__Please read this before doing anything.__**

This folder contains the necessary files to utilize a pyamff calculator with [LAMMPS](https://docs.lammps.org/Manual.html). That is, it allows us the ability to add pyamff as a library to LAMMPS and have LAMMPS and PyAMFF communicate such that PyAMFF can be used as a pair_style. Of course, the pair_style implementations is still a work in progress.

**NOTE**: THIS IS A WORK IN PROGRESS. Certain LAMMPS functionality will not work with the interface as is:

1. This will not allow LAMMPS to automatically download, install, and build pyamff if you haven't do so. You must clone the pyamff (or do `git pull`) manually in order to use it.
2. Cmake options for LAMMPS aren't currently supprted. All compilation must be done using GNU make.
3. I'm sure there are others, which will be dealt with in order of importance or appearance.

To get a copy of lammps, please do

```bash
git clone -b [branch] https://github.com/lammps/lammps.git mylammps
```

in the directory of your choosing, where `[branch]` is replaced with one of the following:

* stable
* release
* develop

Please visit [https://docs.lammps.org/Install_git.html](https://docs.lammps.org/Install_git.html) for more information about the differences between the branches. The following instructions are based on installing on Linux machines. Mac is very similar, and Windows is very different. Instructions for these will be provided later.

**Instructions for Linux**
==========================================
*Setting Environment Variables*
----------------------------------

Once this LAMMPS is cloned, the user should specify the root of the lammps directory as an environment variable in their `.bashrc`:
```bash
export LAMMPS_DIR=[LAMMPS_DIR]
```

where `[LAMMPS_DIR]` is replaced with the location of the directory of your LAMMPS clone (e.g. $HOME/mylammps)
or add it to the .bashrc and run:
```bash
source ~/.bashrc
```

*Compiling PyAMFF for LAMMPS*
----------------------------------
The variable must be called `LAMMPS_DIR`. Nothing else is currently accepted. Now, we move to the root of the pyamff directory. So, run

```bash
cd [PYAMFF_DIR]/pyamff
make LAMMPS=1
```

where `[PYAMFF_DIR] `is replaced with your the location your [PyAMFF] clone (e.g.`$HOME/pyamff`)

*No Environment Variables For ME*
---------------------------------
If you completed the "Setting Environment Variables" section, please go to the next section. If not, keep reading.
If the user is unable to or chose not to set environment variables, the following message will print:

``LAMMPS_DIR undefined. You will need to copy ML-PYAMFF to src manually``

This just means the user needs to do the following:

```bash
cd c_interface
cp -r ML-PYAMFF [LAMMPS_DIR]/src
cd [LAMMPS_DIR]
mkdir -p lib/pyamff
find src/ML-PYAMFF -type l -exec mv {} lib/pyamff/ \;
```

where `[LAMMPS_DIR] `is replaced with your the location of the root of your LAMMPS clone (e.g.`$HOME/mylammps`)

*Inform LAMMPS of PyAMFF*
----------------------------
Now, the user is almost ready to use pyamff with LAMMPS (in theory, pair_style still not fully implemented). The following commands need to be run to link the PyAMFF files and libraries.

```bash
cd [YOUR_LAMMPS_ROOT_DIR]/src
make yes-ml-pyamff
```

After this, LAMMPS can be used as normal.

*Updating PyAMFF interface for LAMMPS*
------------------------------------------
**NOTE**: The libraries are static. Therefore, if the user makes changes to their pyamff or updates it with a `git pull`, the LAMMPS libraries will need to be updated. The user should run

```bash
cd [PYAMFF_DIR]/pyamff
make lammps
cd [LAMMPS_DIR]/src
make package-update
```

where `[LAMMPS_DIR] ` and `[PYAMFF_DIR]` are replaced with the appropriate locations (e.g.`$HOME/mylammps` and`$HOME/pyamff` respectively).

*More Information*
----------------------
For more information about lammps, consider looking at the following resources:

* [https://docs.lammps.org/Manual.html](https://docs.lammps.org/Manual.html)
* [https://www.lammps.org/#gsc.tab=0](https://www.lammps.org/#gsc.tab=0)

