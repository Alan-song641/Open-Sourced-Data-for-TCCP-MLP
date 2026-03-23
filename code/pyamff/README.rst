PyAMFF: Python Atom-Centered Machine Learning Force Field
=========================================================

PyAMFF is a set of tools for fitting and using atomistic machine learning potentials 

Webpage: https://pyamff.gitlab.io/pyamff/index.html

Requirements:
-------------
* Python_ 3.7 or later
* PyTorch_ 1.9.0 or later
* ASE_ 
* NumPy_
* PyYAML_
* Pickle_


Optional:
---------
* gcc-gfortran_


Installation:
-------------
* Using git with ssh::

    $ git clone git@gitlab.com:pyamff/pyamff.git

* Using git with HTML::

    $ git clone https://gitlab.com/pyamff/pyamff.git

* To build fortran modules::

    $ cd path/to/pyamff/pyamff/ 
    $ make

* Make sure to add /path/to/pyamff/bin to your $PATH and /path/to/pyamff/ to your $PYTHONPATH

.. note::
    After installation, you can go to /pyammf/tests/ and run *run_tests.sh* to make sure the build was successful



Contact:
-------

* Henkelman Group (UT Austin)
* Lei Li Group (SUSTC)


.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _ASE: https://gitlab.com/ase/ase
.. _Torch: https://pytorch.org/
.. _PyYAML: https://pypi.org/project/PyYAML/
.. _Pickle: https://pypi.org/project/pickle-mixin/
.. _gcc-gfortran: https://gcc.gnu.org/wiki/GFortran

