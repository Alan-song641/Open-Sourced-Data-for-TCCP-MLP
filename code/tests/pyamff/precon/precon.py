"""
Implementation of the Precon abstract base class and subclasses
"""
import time
from itertools import product
from numpy.linalg import eigh
import scipy.linalg as la
import numpy as np
from scipy import sparse, rand
from scipy.sparse.linalg import spsolve
from ase.units import Hartree, Bohr
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import wrap_positions
import ase.utils.ff as ff
import ase.units as units
from pyamff.precon import logger
from pyamff.fingerprints.behlerParrinello import loop_FPs_precon
from pyamff.neighborlist import NeighborLists
from pyamff.config import ConfigClass
from collections import OrderedDict
from pyamff.precon.neighbors import (get_neighbours, have_matscipy, estimate_nearest_neighbour_distance)
try:
    from pyamg import smoothed_aggregation_solver
    have_pyamg = True
except ImportError:
    have_pyamg = False

THz = 1e12 * 1. / units.s


class Precon(object):

    def __init__(self, r_cut=None, r_NN=None,
                 mu=None, mu_c=None,
                 dim=3, c_stab=0.1, force_stab=False,
                 recalc_mu=False, array_convention='C',
                 use_pyamg=True, solve_tol=1e-8,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False):
        """Initialise a preconditioner object based on passed parameters.

        Args:
            r_cut: float. This is a cut-off radius. The preconditioner matrix
                will be created by considering pairs of atoms that are within a
                distance r_cut of each other. For a regular lattice, this is
                usually taken somewhere between the first- and second-nearest
                neighbour distance. If r_cut is not provided, default is
                2 * r_NN (see below)
            r_NN: nearest neighbour distance. If not provided, this is
                  calculated
                from input structure.
            mu: float
                energy scale for position degreees of freedom. If `None`, mu
                is precomputed using finite difference derivatives.
            mu_c: float
                energy scale for cell degreees of freedom. Also precomputed
                if None.
            estimate_mu_eigmode:
                If True, estimates mu based on the lowest eigenmodes of
                unstabilised preconditioner. If False it uses the sine based
                approach.
            dim: int; dimensions of the problem
            c_stab: float. The diagonal of the preconditioner matrix will have
                a stabilisation constant added, which will be the value of
                c_stab times mu.
            force_stab:
                If True, always add the stabilisation to diagnonal, regardless
                of the presence of fixed atoms.
            recalc_mu: if True, the value of mu will be recalculated every time
                self.make_precon is called. This can be overridden in specific
                cases with recalc_mu argument in self.make_precon. If recalc_mu
                is set to True here, the value passed for mu will be
                irrelevant unless recalc_mu is set False the first time
                make_precon is called.
            array_convention: Either 'C' or 'F' for Fortran; this will change
                the preconditioner to reflect the ordering of the indices in
                the vector it will operate on. The C convention assumes the
                vector will be arranged atom-by-atom (ie [x1, y1, z1, x2, ...])
                while the F convention assumes it will be arranged component
                by component (ie [x1, x2, ..., y1, y2, ...]).
            use_pyamg: use PyAMG to solve P x = y, if available.
            solve_tol: tolerance used for PyAMG sparse linear solver,
            if available.
            apply_positions: if True, apply preconditioner to position DoF
            apply_cell: if True, apply preconditioner to cell DoF

        Raises:
            ValueError for problem with arguments

        """

        self.r_NN = r_NN
        self.r_cut = r_cut
        self.mu = mu
        self.mu_c = mu_c
        self.estimate_mu_eigmode = estimate_mu_eigmode
        self.c_stab = c_stab
        self.force_stab = force_stab
        self.array_convention = array_convention
        self.recalc_mu = recalc_mu
        self.P = None
        self.old_positions = None

        if use_pyamg and not have_pyamg:
            use_pyamg = False
            logger.warning('use_pyamg=True but PyAMG cannot be imported! '
                           'falling back on direct inversion of '
                           'preconditioner, may be slow for large systems')

        self.use_pyamg = use_pyamg
        self.solve_tol = solve_tol
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell

        if dim < 1:
            raise ValueError('Dimension must be at least 1')
        self.dim = dim

        if not have_matscipy:
            logger.info('Unable to import Matscipy. Neighbour list '
                        'calculations may be very slow.')

    def make_precon(self, atoms, recalc_mu=None):
        """Create a preconditioner matrix based on the passed set of atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned.

        Args:
            atoms: the Atoms object used to create the preconditioner.
                Can also
            recalc_mu: if True, self.mu (and self.mu_c for variable cell)
                will be recalculated by calling self.estimate_mu(atoms)
                before the preconditioner matrix is created. If False, self.mu
                will be calculated only if it does not currently have a value
                (ie, the first time this function is called).

        Returns:
            A two-element tuple:
                P: A sparse scipy csr_matrix. BE AWARE that using
                    numpy.dot() with sparse matrices will result in
                    errors/incorrect results - use the .dot method directly
                    on the matrix instead.
        """

        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms)

        if self.r_cut is None:
            # This is the first time this function has been called, and no
            # cutoff radius has been specified, so calculate it automatically.
            self.r_cut = 2.0 * self.r_NN
        elif self.r_cut < self.r_NN:
            warning = ('WARNING: r_cut (%.2f) < r_NN (%.2f), '
                       'increasing to 1.1*r_NN = %.2f' % (self.r_cut,
                                                          self.r_NN,
                                                          1.1 * self.r_NN))
            logger.info(warning)
            print(warning)
            self.r_cut = 1.1 * self.r_NN
        print('RCUT', self.r_cut)

        if recalc_mu is None:
            # The caller has not specified whether or not to recalculate mu,
            # so the Precon's setting is used.
            recalc_mu = self.recalc_mu

        if self.mu is None:
            # Regardless of what the caller has specified, if we don't
            # currently have a value of mu, then we need one.
            recalc_mu = True

        if recalc_mu:
            self.estimate_mu(atoms)

        if self.P is not None:
            real_atoms = atoms
            if isinstance(atoms, Filter):
                real_atoms = atoms.atoms
            if self.old_positions is None:
                self.old_positions = wrap_positions(real_atoms.positions,
                                                    real_atoms.cell)
            displacement = wrap_positions(real_atoms.positions,
                                          real_atoms.cell) - self.old_positions
            self.old_positions = real_atoms.get_positions()
            max_abs_displacement = abs(displacement).max()
            logger.info('max(abs(displacements)) = %.2f A (%.2f r_NN)',
                        max_abs_displacement, max_abs_displacement / self.r_NN)
            if max_abs_displacement < 0.5 * self.r_NN:
                return self.P

        start_time = time.time()

        # Create the preconditioner:
        self._make_sparse_precon(atoms, force_stab=self.force_stab)

        logger.info('--- Precon created in %s seconds ---',
                    time.time() - start_time)
        return self.P

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """create a sparse preconditioner matrix based on the passed atoms.

        creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. the matrix will be stored in the attribute self.p and
        returned. note that this function will use self.mu, whatever it is.

        args:
            atoms: the atoms object used to create the preconditioner.

        returns:
            a scipy.sparse.csr_matrix object, representing a d*n by d*n matrix
            (where n is the number of atoms, and d is the value of self.dim).
            be aware that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        logger.info('creating sparse precon: initial_assembly=%r, '
                    'force_stab=%r, apply_positions=%r, apply_cell=%r',
                    initial_assembly, force_stab, self.apply_positions,
                    self.apply_cell)

        n = len(atoms)
        diag_i = np.arange(n, dtype=int)
        start_time = time.time()
        if self.apply_positions:
            # compute neighbour list
            i, j, rij, fixed_atoms = get_neighbours(atoms, self.r_cut)
            logger.info('--- neighbour list created in %s s ---' %
                        (time.time() - start_time))
#            print('i',i)
#            print('j', j)
#            print('rij', rij)

            # compute entries in triplet format: without the constraints
            start_time = time.time()
            coeff = self.get_coeff(rij)
            print('coeff len', len(coeff))
#            print("i", i)
            print('coeff', coeff)
#            print('n', n)
            diag_coeff = np.bincount(i, -coeff, minlength=n).astype(np.float64)
            print('dc',diag_coeff)
#            print('diag_coeff', diag_coeff)
            if force_stab or len(fixed_atoms) == 0:
                logger.info('adding stabilisation to preconditioner')
                diag_coeff += self.mu * self.c_stab
#                diag_coeff = -diag_coeff
#                print(diag_coeff)

#                print('diag forcestab', diag_coeff)
        else:
            diag_coeff = np.ones(n)

        # precon is mu_c*identity for cell dof
        if isinstance(atoms, filter):
            if self.apply_cell:
                diag_coeff[-3] = self.mu_c
                diag_coeff[-2] = self.mu_c
                diag_coeff[-1] = self.mu_c
            else:
                diag_coeff[-3] = 1.0
                diag_coeff[-2] = 1.0
                diag_coeff[-1] = 1.0
        logger.info('--- computed triplet format in %s s ---' %
                    (time.time() - start_time))

        if self.apply_positions and not initial_assembly:
            # apply the constraints
            start_time = time.time()
            mask = np.ones(n)
            mask[fixed_atoms] = 0.0
            coeff *= mask[i] * mask[j]
            diag_coeff[fixed_atoms] = 1.0
            logger.info('--- applied fixed_atoms in %s s ---' %
                        (time.time() - start_time))

        if self.apply_positions:
            # remove zeros
            start_time = time.time()
            inz = np.nonzero(coeff)
            i = np.hstack((i[inz], diag_i))
            j = np.hstack((j[inz], diag_i))
            coeff = np.hstack((coeff[inz], diag_coeff))
            logger.info('--- remove zeros in %s s ---' %
                        (time.time() - start_time))
        else:
            i = diag_i
            j = diag_i
            coeff = diag_coeff

        # create the matrix
        start_time = time.time()
        csc_p = sparse.csc_matrix((coeff, (i, j)), shape=(n, n))
        logger.info('--- created csc matrix in %s s ---' %
                    (time.time() - start_time))

        self.csc_p = csc_p

        start_time = time.time()
        if self.dim == 1:
            self.p = csc_p
        elif self.array_convention == 'f':
            csc_p = csc_p.tocsr()
            self.p = csc_p
            for i in range(self.dim - 1):
                self.p = sparse.block_diag((self.p, csc_p)).tocsr()
        else:
            # convert back to triplet and read the arrays
            csc_p = csc_p.tocoo()
            i = csc_p.row * self.dim
            j = csc_p.col * self.dim
            z = csc_p.data

            # n-dimensionalise, interlaced coordinates
            i = np.hstack([i + d for d in range(self.dim)])
            j = np.hstack([j + d for d in range(self.dim)])
            z = np.hstack([z for d in range(self.dim)])
            self.p = sparse.csc_matrix((z, (i, j)),
                                       shape=(self.dim * n, self.dim * n))
            self.p = self.p.tocsr()
        logger.info('--- n-dim precon created in %s s ---' %
                    (time.time() - start_time))

        # create solver
        if self.use_pyamg and have_pyamg:
            start_time = time.time()
            self.ml = smoothed_aggregation_solver(
                self.p, b=none,
                strength=('symmetric', {'theta': 0.0}),
                smooth=(
                    'jacobi', {'filter': true, 'weighting': 'local'}),
                improve_candidates=[('block_gauss_seidel',
                                     {'sweep': 'symmetric', 'iterations': 4}),
                                    none, none, none, none, none, none, none,
                                    none, none, none, none, none, none, none],
                aggregate='standard',
                presmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver='pinv')
            logger.info('--- multi grid solver created in %s s ---' %
                        (time.time() - start_time))
        print("p", self.p)
#        print('parr', self.p.toarray())
#        print('len p',self.p.count_nonzero)
#        print('type', type(self.p))
        return self.p

    def dot(self, x, y):
        """
        Return the preconditioned dot product <P x, y>

        Uses 128-bit floating point math for vector dot products
        """
        return longsum(self.P.dot(x) * y)

    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        start_time = time.time()
        if self.use_pyamg and have_pyamg:
            y = self.ml.solve(x, x0=rand(self.P.shape[0]),
                              tol=self.solve_tol,
                              accel='cg',
                              maxiter=300,
                              cycle='W')
        else:
            y = spsolve(self.P, x)
        logger.info('--- Precon applied in %s seconds ---',
                    time.time() - start_time)
        return y

    def get_coeff(self, r):
        raise NotImplementedError('Must be overridden by subclasses')

    def estimate_mu(self, atoms, H=None):
        """
        Estimate optimal preconditioner coefficient \mu

        \mu is estimated from a numerical solution of

            [dE(p+v) -  dE(p)] \cdot v = \mu < P1 v, v >

        with perturbation

            v(x,y,z) = H P_lowest_nonzero_eigvec(x, y, z)

            or

            v(x,y,z) = H (sin(x / Lx), sin(y / Ly), sin(z / Lz))

        After the optimal \mu is found, self.mu will be set to its value.

        If `atoms` is an instance of Filter an additional \mu_c
        will be computed for the cell degrees of freedom .

        Args:
            atoms: Atoms object for initial system

            H: 3x3 array or None
                Magnitude of deformation to apply.
                Default is 1e-2*rNN*np.eye(3)

        Returns:
            mu   : float
            mu_c : float or None
        """

        if self.dim != 3:
            raise ValueError('Automatic calculation of mu only possible for '
                             'three-dimensional preconditioners. Try setting '
                             'mu manually instead.')

        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms)

        # deformation matrix, default is diagonal
        if H is None:
            H = 1e-2 * self.r_NN * np.eye(3)

        # compute perturbation
        p = atoms.get_positions()

        if self.estimate_mu_eigmode:
            self.mu = 1.0
            self.mu_c = 1.0
            c_stab = self.c_stab
            self.c_stab = 0.0

            if isinstance(atoms, Filter):
                n = len(atoms.atoms)
            else:
                n = len(atoms)
            P0 = self._make_sparse_precon(atoms,
                                          initial_assembly=True)[:3 * n,
                                                                 :3 * n]
            eigvals, eigvecs = sparse.linalg.eigsh(P0, k=4, which='SM')

            logger.debug('estimate_mu(): lowest 4 eigvals = %f %f %f %f'
                         % (eigvals[0], eigvals[1], eigvals[2], eigvals[3]))
            # check eigenvalues
            if any(eigvals[0:3] > 1e-6):
                raise ValueError('First 3 eigenvalues of preconditioner matrix'
                                 'do not correspond to translational modes.')
            elif eigvals[3] < 1e-6:
                raise ValueError('Fourth smallest eigenvalue of '
                                 'preconditioner matrix '
                                 'is too small, increase r_cut.')

            x = np.zeros(n)
            for i in range(n):
                x[i] = eigvecs[:, 3][3 * i]
            x = x / np.linalg.norm(x)
            if x[0] < 0:
                x = -x

            v = np.zeros(3 * len(atoms))
            for i in range(n):
                v[3 * i] = x[i]
                v[3 * i + 1] = x[i]
                v[3 * i + 2] = x[i]
            v = v / np.linalg.norm(v)
            v = v.reshape((-1, 3))

            self.c_stab = c_stab
        else:
            Lx, Ly, Lz = [p[:, i].max() - p[:, i].min() for i in range(3)]
            logger.debug('estimate_mu(): Lx=%.1f Ly=%.1f Lz=%.1f',
                         Lx, Ly, Lz)

            x, y, z = p.T
            # sine_vr = [np.sin(x/Lx), np.sin(y/Ly), np.sin(z/Lz)], but we need
            # to take into account the possibility that one of Lx/Ly/Lz is
            # zero.
            sine_vr = [x, y, z]

            for i, L in enumerate([Lx, Ly, Lz]):
                if L == 0:
                    logger.warning(
                        'Cell length L[%d] == 0. Setting H[%d,%d] = 0.' %
                        (i, i, i))
                    H[i, i] = 0.0
                else:
                    sine_vr[i] = np.sin(sine_vr[i] / L)

            v = np.dot(H, sine_vr).T

        natoms = len(atoms)
        if isinstance(atoms, Filter):
            natoms = len(atoms.atoms)
            eps = H / self.r_NN
            v[natoms:, :] = eps

        v1 = v.reshape(-1)

        # compute LHS
        dE_p = -atoms.get_forces().reshape(-1)
        atoms_v = atoms.copy()
        atoms_v.set_calculator(atoms.get_calculator())
        if isinstance(atoms, Filter):
            atoms_v = atoms.__class__(atoms_v)
            if hasattr(atoms, 'constant_volume'):
                atoms_v.constant_volume = atoms.constant_volume
        atoms_v.set_positions(p + v)
        dE_p_plus_v = -atoms_v.get_forces().reshape(-1)

        # compute left hand side
        LHS = (dE_p_plus_v - dE_p) * v1

        # assemble P with \mu = 1
        self.mu = 1.0
        self.mu_c = 1.0

        P1 = self._make_sparse_precon(atoms, initial_assembly=True)

        # compute right hand side
        RHS = P1.dot(v1) * v1

        # use partial sums to compute separate mu for positions and cell DoFs
        self.mu = longsum(LHS[:3 * natoms]) / longsum(RHS[:3 * natoms])
        if self.mu < 1.0:
            logger.info('mu (%.3f) < 1.0, capping at mu=1.0', self.mu)
            self.mu = 1.0

        if isinstance(atoms, Filter):
            self.mu_c = longsum(LHS[3 * natoms:]) / longsum(RHS[3 * natoms:])
            if self.mu_c < 1.0:
                logger.info(
                    'mu_c (%.3f) < 1.0, capping at mu_c=1.0', self.mu_c)
                self.mu_c = 1.0

        logger.info('estimate_mu(): mu=%r, mu_c=%r', self.mu, self.mu_c)

        self.P = None  # force a rebuild with new mu (there may be fixed atoms)
        return (self.mu, self.mu_c)


class Pfrommer(object):
    """Use initial guess for inverse Hessian from Pfrommer et al. as a
    simple preconditioner

    J. Comput. Phys. vol 131 p233-240 (1997)
    """

    def __init__(self, bulk_modulus=500 * units.GPa, phonon_frequency=50 * THz,
                 apply_positions=True, apply_cell=True):
        """
        Default bulk modulus is 500 GPa and default phonon frequency is 50 THz
        """

        self.bulk_modulus = bulk_modulus
        self.phonon_frequency = phonon_frequency
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell
        self.H0 = None

    def make_precon(self, atoms):
        if self.H0 is not None:
            # only build H0 on first call
            return NotImplemented

        variable_cell = False
        if isinstance(atoms, Filter):
            variable_cell = True
            atoms = atoms.atoms

        # position DoF
        omega = self.phonon_frequency
        mass = atoms.get_masses().mean()
        block = np.eye(3) / (mass * omega**2)
        blocks = [block] * len(atoms)

        # cell DoF
        if variable_cell:
            coeff = 1.0
            if self.apply_cell:
                coeff = 1.0 / (3 * self.bulk_modulus)
            blocks.append(np.diag([coeff] * 9))

        self.H0 = sparse.block_diag(blocks, format='csr')
        return NotImplemented

    def dot(self, x, y):
        """
        Return the preconditioned dot product <P x, y>

        Uses 128-bit floating point math for vector dot products
        """
        raise NotImplementedError

    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        y = self.H0.dot(x)
        return y


class C1(Precon):
    """Creates matrix by inserting a constant whenever r_ij is less than r_cut.
    """

    def __init__(self, r_cut=None, mu=None, mu_c=None, dim=3, c_stab=0.1,
                 force_stab=False,
                 recalc_mu=False, array_convention='C',
                 use_pyamg=True, solve_tol=1e-9,
                 apply_positions=True, apply_cell=True):
        Precon.__init__(self, r_cut=r_cut, mu=mu, mu_c=mu_c,
                        dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        use_pyamg=use_pyamg, solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell)

    def get_coeff(self, r):
        return -self.mu * np.ones_like(r)

class G1(Precon):
    """Creates matrix by inserting a constant whenever r_ij is less than r_cut.
    """

    def __init__(self, eta=120., Rs = 0.5, Rcut = 6., r_cut=None, mu=None, mu_c=None, dim=3, c_stab=0.1,
                 force_stab=False,
                 recalc_mu=False, array_convention='C',
                 use_pyamg=True, solve_tol=1e-9,
                 apply_positions=True, apply_cell=True):
        Precon.__init__(self, r_cut=r_cut, mu=mu, mu_c=mu_c,
                        dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        use_pyamg=use_pyamg, solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell)
        self.eta = eta
        self.Rs = Rs
        self.Rcut = Rcut

    def get_coeff(self, r):
        print('mu',self.mu)
        return np.exp(-self.eta * ((r - self.Rs)**2)/(self.Rcut**2)) * (0.5* (np.cos(np.pi * r / self.Rcut) + 1)) 

class G1P(Precon):
    """Creates matrix by inserting a constant whenever r_ij is less than r_cut.
    """

    def __init__(self, eta=120., Rs = 0.5, Rcut = 6., r_cut=None, mu=None, mu_c=None, dim=3, c_stab=0.1,
                 force_stab=False,
                 recalc_mu=False, array_convention='C',
                 use_pyamg=True, solve_tol=1e-9,
                 apply_positions=True, apply_cell=True):
        Precon.__init__(self, r_cut=r_cut, mu=mu, mu_c=mu_c,
                        dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        use_pyamg=use_pyamg, solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell)
        self.eta = eta
        self.Rs = Rs
        self.Rcut = Rcut

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """create a sparse preconditioner matrix based on the passed atoms.

        creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. the matrix will be stored in the attribute self.p and
        returned. note that this function will use self.mu, whatever it is.

        args:
            atoms: the atoms object used to create the preconditioner.

        returns:
            a scipy.sparse.csr_matrix object, representing a d*n by d*n matrix
            (where n is the number of atoms, and d is the value of self.dim).
            be aware that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        print('mu',self.mu)
        config = ConfigClass()
        config.initialize()

        nl = NeighborLists(cutoff=5.847888733346419)
        images=OrderedDict()
        images[0] = atoms
        nl.calculate(images,fortran=False)
        fp_paras = config.config['fp_paras'].fp_paras
        fps, fpprimes = loop_FPs_precon(0,nl, fp_paras, fortran=False)
        N=len(atoms)
        p = np.zeros((N*3, N*3))
        
        for center in fps:
            for neighbor in fps[center][0]:
                p[center][neighbor] = fps[center][0][neighbor]
        Psparse = sparse.csr_matrix(p)
#        print('Psparse', Psparse)

        #for key, value in fpprimes:
            #p[key][value] = -fpprimes[key,value][0]
#
#        Psparse = sparse.csr_matrix(p)
#        print(Psparse)
#        print(self.mu)
#        print('lenPsparse', len(Psparse.nonzero()[0]))
        diag = Psparse.diagonal()
#        print(diag)
        diag += self.mu* self.c_stab
#        print('d+m',diag)
        Psparse.setdiag(diag)
#        Psparse.eliminate_zeros()
#        P = Psparse.data #Psparse[Psparse.nonzero()]
#        P = p.flatten()
        self.P = Psparse
        print(self.P)

#        print('parr', self.p.toarray())
#        print('len p',self.p.count_nonzero)
#        print('type', type(self.p))
        return self.P


    def get_coeff(self, atoms):
        print('mu',self.mu)
        config = ConfigClass()
        config.initialize()

        nl = NeighborLists(cutoff=5.847888733346419)
        images=OrderedDict()
        images[0] = atoms
        nl.calculate(images,fortran=False)
        fp_paras = config.config['fp_paras'].fp_paras
        fps, fpprimes = loop_FPs_precon(0,nl, fp_paras, fortran=False)
        N=len(atoms)
        p = np.zeros((N*3, N*3))
        
        for center in fps:
            for neighbor in fps[center][0]:
                p[center][neighbor] = fps[center][0][neighbor]
        Psparse = sparse.csr_matrix(p)
        print('Psparse', Psparse)

        #for key, value in fpprimes:
            #p[key][value] = -fpprimes[key,value][0]
#
#        Psparse = sparse.csr_matrix(p)
#        print(Psparse)
#        print(self.mu)
#        print('lenPsparse', len(Psparse.nonzero()[0]))
#        diag = Psparse.diagonal()
#        diag += self.mu
        Psparse.setdiag(0)
        Psparse.eliminate_zeros()
        P = Psparse.data #Psparse[Psparse.nonzero()]
#        P = p.flatten()
#        self.P = Psparse



        return P



class Exp(Precon):
    """Creates matrix with values decreasing exponentially with distance.
    """

    def __init__(self, A=3.0, r_cut=None, r_NN=None, mu=None, mu_c=None,
                 dim=3, c_stab=0.1,
                 force_stab=False, recalc_mu=False, array_convention='C',
                 use_pyamg=True, solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False):
        """Initialise an Exp preconditioner with given parameters.

        Args:
            r_cut, mu, c_stab, dim, sparse, recalc_mu, array_convention: see
                precon.__init__()
            A: coefficient in exp(-A*r/r_NN). Default is A=3.0.
        """
        Precon.__init__(self, r_cut=r_cut, r_NN=r_NN,
                        mu=mu, mu_c=mu_c, dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        use_pyamg=use_pyamg,
                        solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell,
                        estimate_mu_eigmode=estimate_mu_eigmode)

        self.A = A

    def get_coeff(self, r):
        print(self.P)
        print(self.mu)
        return -self.mu * np.exp(-self.A * (r / self.r_NN - 1))
        

class FF(Precon):
    """Creates matrix using morse/bond/angle/dihedral force field parameters.
    """

    def __init__(self, dim=3, c_stab=0.1, force_stab=False,
                 array_convention='C', use_pyamg=True, solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 hessian='reduced', morses=None, bonds=None, angles=None,
                 dihedrals=None):
        """Initialise an FF preconditioner with given parameters.

        Args:
             dim, c_stab, force_stab, array_convention: see
             precon.__init__(), use_pyamg, solve_tol
             morses: class Morse
             bonds: class Bond
             angles: class Angle
             dihedrals: class Dihedral
        """

        if (morses is None and bonds is None and angles is None and
            dihedrals is None):
            raise ImportError(
                'At least one of morses, bonds, angles or dihedrals must be '
                'defined!')

        Precon.__init__(self,
                        dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        array_convention=array_convention,
                        use_pyamg=use_pyamg,
                        solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell)

        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms):

        start_time = time.time()

        # Create the preconditioner:
        self._make_sparse_precon(atoms, force_stab=self.force_stab)

        logger.info('--- Precon created in %s seconds ---',
                    time.time() - start_time)
        return self.P

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """ """

        start_time = time.time()

        N = len(atoms)

        row = []
        col = []
        data = []

        if self.morses is not None:

            for n in range(len(self.morses)):
                if self.hessian == 'reduced':
                    i, j, Hx = ff.get_morse_potential_reduced_hessian(
                        atoms, self.morses[n])
                elif self.hessian == 'spectral':
                    i, j, Hx = ff.get_morse_potential_hessian(
                        atoms, self.morses[n], spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
                row.extend(np.repeat(x, 6))
                col.extend(np.tile(x, 6))
                data.extend(Hx.flatten())

        if self.bonds is not None:

            for n in range(len(self.bonds)):
                if self.hessian == 'reduced':
                    i, j, Hx = ff.get_bond_potential_reduced_hessian(
                        atoms, self.bonds[n], self.morses)
                elif self.hessian == 'spectral':
                    i, j, Hx = ff.get_bond_potential_hessian(
                        atoms, self.bonds[n], self.morses, spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
                row.extend(np.repeat(x, 6))
                col.extend(np.tile(x, 6))
                data.extend(Hx.flatten())

        if self.angles is not None:

            for n in range(len(self.angles)):
                if self.hessian == 'reduced':
                    i, j, k, Hx = ff.get_angle_potential_reduced_hessian(
                        atoms, self.angles[n], self.morses)
                elif self.hessian == 'spectral':
                    i, j, k, Hx = ff.get_angle_potential_hessian(
                        atoms, self.angles[n], self.morses, spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 *
                     j + 1, 3 * j + 2, 3 * k, 3 * k + 1, 3 * k + 2]
                row.extend(np.repeat(x, 9))
                col.extend(np.tile(x, 9))
                data.extend(Hx.flatten())

        if self.dihedrals is not None:

            for n in range(len(self.dihedrals)):
                if self.hessian == 'reduced':
                    i, j, k, l, Hx = \
                        ff.get_dihedral_potential_reduced_hessian(
                            atoms, self.dihedrals[n], self.morses)
                elif self.hessian == 'spectral':
                    i, j, k, l, Hx = ff.get_dihedral_potential_hessian(
                        atoms, self.dihedrals[n], self.morses, spectral=True)
                else:
                    raise NotImplementedError('Not implemented hessian')
                x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j +
                     2, 3 * k, 3 * k + 1, 3 * k + 2, 3 * l, 3 * l + 1,
                     3 * l + 2]
                row.extend(np.repeat(x, 12))
                col.extend(np.tile(x, 12))
                data.extend(Hx.flatten())

        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        data.extend([self.c_stab] * self.dim * N)

        # create the matrix
        start_time = time.time()
        self.P = sparse.csc_matrix(
            (data, (row, col)), shape=(self.dim * N, self.dim * N))
        logger.info('--- created CSC matrix in %s s ---' %
                    (time.time() - start_time))

        fixed_atoms = []
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_atoms.extend(list(constraint.index))
            else:
                raise TypeError(
                    'only FixAtoms constraints are supported by Precon class')
        if len(fixed_atoms) != 0:
            self.P.tolil()
        for i in fixed_atoms:
            self.P[i, :] = 0.0
            self.P[:, i] = 0.0
            self.P[i, i] = 1.0

        self.P = self.P.tocsr()

        logger.info('--- N-dim precon created in %s s ---' %
                    (time.time() - start_time))

        # Create solver
        if self.use_pyamg and have_pyamg:
            start_time = time.time()
            self.ml = smoothed_aggregation_solver(
                self.P, B=None,
                strength=('symmetric', {'theta': 0.0}),
                smooth=(
                    'jacobi', {'filter': True, 'weighting': 'local'}),
                improve_candidates=[('block_gauss_seidel',
                                     {'sweep': 'symmetric', 'iterations': 4}),
                                    None, None, None, None, None, None, None,
                                    None, None, None, None, None, None, None],
                aggregate='standard',
                presmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver='pinv')
            logger.info('--- multi grid solver created in %s s ---' %
                        (time.time() - start_time))

        return self.P


class Exp_FF(Exp, FF):
    """Creates matrix with values decreasing exponentially with distance.
    """

    def __init__(self, A=3.0, r_cut=None, r_NN=None, mu=None, mu_c=None,
                 dim=3, c_stab=0.1,
                 force_stab=False, recalc_mu=False, array_convention='C',
                 use_pyamg=True, solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False,
                 hessian='reduced', morses=None, bonds=None, angles=None,
                 dihedrals=None):
        """Initialise an Exp+FF preconditioner with given parameters.

        Args:
            r_cut, mu, c_stab, dim, recalc_mu, array_convention: see
                precon.__init__()
            A: coefficient in exp(-A*r/r_NN). Default is A=3.0.
        """
        if (morses is None and bonds is None and angles is None and
            dihedrals is None):
            raise ImportError(
                'At least one of morses, bonds, angles or dihedrals must '
                'be defined!')

        Precon.__init__(self, r_cut=r_cut, r_NN=r_NN,
                        mu=mu, mu_c=mu_c, dim=dim, c_stab=c_stab,
                        force_stab=force_stab,
                        recalc_mu=recalc_mu,
                        array_convention=array_convention,
                        use_pyamg=use_pyamg,
                        solve_tol=solve_tol,
                        apply_positions=apply_positions,
                        apply_cell=apply_cell,
                        estimate_mu_eigmode=estimate_mu_eigmode)

        self.A = A
        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms, recalc_mu=None):

        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms)

        if self.r_cut is None:
            # This is the first time this function has been called, and no
            # cutoff radius has been specified, so calculate it automatically.
            self.r_cut = 2.0 * self.r_NN
        elif self.r_cut < self.r_NN:
            warning = ('WARNING: r_cut (%.2f) < r_NN (%.2f), '
                       'increasing to 1.1*r_NN = %.2f' % (self.r_cut,
                                                          self.r_NN,
                                                          1.1 * self.r_NN))
            logger.info(warning)
            print(warning)
            self.r_cut = 1.1 * self.r_NN

        if recalc_mu is None:
            # The caller has not specified whether or not to recalculate mu,
            # so the Precon's setting is used.
            recalc_mu = self.recalc_mu

        if self.mu is None:
            # Regardless of what the caller has specified, if we don't
            # currently have a value of mu, then we need one.
            recalc_mu = True

        if recalc_mu:
            self.estimate_mu(atoms)

        if self.P is not None:
            real_atoms = atoms
            if isinstance(atoms, Filter):
                real_atoms = atoms.atoms
            if self.old_positions is None:
                self.old_positions = wrap_positions(real_atoms.positions,
                                                    real_atoms.cell)
            displacement = wrap_positions(real_atoms.positions,
                                          real_atoms.cell) - self.old_positions
            self.old_positions = real_atoms.get_positions()
            max_abs_displacement = abs(displacement).max()
            logger.info('max(abs(displacements)) = %.2f A (%.2f r_NN)',
                        max_abs_displacement,
                        max_abs_displacement / self.r_NN)
            if max_abs_displacement < 0.5 * self.r_NN:
                return self.P

        start_time = time.time()

        # Create the preconditioner:
        self._make_sparse_precon(atoms, force_stab=self.force_stab)

        logger.info('--- Precon created in %s seconds ---',
                    time.time() - start_time)
        return self.P

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        logger.info('creating sparse precon: initial_assembly=%r, '
                    'force_stab=%r, apply_positions=%r, apply_cell=%r',
                    initial_assembly, force_stab, self.apply_positions,
                    self.apply_cell)

        N = len(atoms)
        start_time = time.time()
        if self.apply_positions:
            # compute neighbour list
            i_list, j_list, rij_list, fixed_atoms = get_neighbours(
                atoms, self.r_cut)
            logger.info('--- neighbour list created in %s s ---' %
                        (time.time() - start_time))

        row = []
        col = []
        data = []

        # precon is mu_c*identity for cell DoF
        if isinstance(atoms, Filter):
            i = N - 3
            j = N - 2
            k = N - 1
            x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 *
                 j + 1, 3 * j + 2, 3 * k, 3 * k + 1, 3 * k + 2]
            row.extend(x)
            col.extend(x)
            if self.apply_cell:
                data.extend(np.repeat(self.mu_c, 9))
            else:
                data.extend(np.repeat(self.mu_c, 9))
        logger.info('--- computed triplet format in %s s ---' %
                    (time.time() - start_time))

        conn = sparse.lil_matrix((N, N), dtype=bool)

        if self.apply_positions and not initial_assembly:

            if self.morses is not None:

                for n in range(len(self.morses)):
                    if self.hessian == 'reduced':
                        i, j, Hx = ff.get_morse_potential_reduced_hessian(
                            atoms, self.morses[n])
                    elif self.hessian == 'spectral':
                        i, j, Hx = ff.get_morse_potential_hessian(
                            atoms, self.morses[n], spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2,
                         3 * j, 3 * j + 1, 3 * j + 2]
                    row.extend(np.repeat(x, 6))
                    col.extend(np.tile(x, 6))
                    data.extend(Hx.flatten())
                    conn[i, j] = True
                    conn[j, i] = True

            if self.bonds is not None:

                for n in range(len(self.bonds)):
                    if self.hessian == 'reduced':
                        i, j, Hx = ff.get_bond_potential_reduced_hessian(
                            atoms, self.bonds[n], self.morses)
                    elif self.hessian == 'spectral':
                        i, j, Hx = ff.get_bond_potential_hessian(
                            atoms, self.bonds[n], self.morses, spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2,
                         3 * j, 3 * j + 1, 3 * j + 2]
                    row.extend(np.repeat(x, 6))
                    col.extend(np.tile(x, 6))
                    data.extend(Hx.flatten())
                    conn[i, j] = True
                    conn[j, i] = True

            if self.angles is not None:

                for n in range(len(self.angles)):
                    if self.hessian == 'reduced':
                        i, j, k, Hx = ff.get_angle_potential_reduced_hessian(
                            atoms, self.angles[n], self.morses)
                    elif self.hessian == 'spectral':
                        i, j, k, Hx = ff.get_angle_potential_hessian(
                            atoms, self.angles[n], self.morses, spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 *
                         j + 1, 3 * j + 2, 3 * k, 3 * k + 1, 3 * k + 2]
                    row.extend(np.repeat(x, 9))
                    col.extend(np.tile(x, 9))
                    data.extend(Hx.flatten())
                    conn[i, j] = conn[i, k] = conn[j, k] = True
                    conn[j, i] = conn[k, i] = conn[k, j] = True

            if self.dihedrals is not None:

                for n in range(len(self.dihedrals)):
                    if self.hessian == 'reduced':
                        i, j, k, l, Hx = \
                            ff.get_dihedral_potential_reduced_hessian(
                                atoms, self.dihedrals[n], self.morses)
                    elif self.hessian == 'spectral':
                        i, j, k, l, Hx = ff.get_dihedral_potential_hessian(
                            atoms, self.dihedrals[n], self.morses,
                            spectral=True)
                    else:
                        raise NotImplementedError('Not implemented hessian')
                    x = [3 * i, 3 * i + 1, 3 * i + 2,
                         3 * j, 3 * j + 1, 3 * j + 2,
                         3 * k, 3 * k + 1, 3 * k + 2,
                         3 * l, 3 * l + 1, 3 * l + 2]
                    row.extend(np.repeat(x, 12))
                    col.extend(np.tile(x, 12))
                    data.extend(Hx.flatten())
                    conn[i, j] = conn[i, k] = conn[i, l] = conn[
                        j, k] = conn[j, l] = conn[k, l] = True
                    conn[j, i] = conn[k, i] = conn[l, i] = conn[
                        k, j] = conn[l, j] = conn[l, k] = True

        if self.apply_positions:
            for i, j, rij in zip(i_list, j_list, rij_list):
                if not conn[i, j]:
                    coeff = self.get_coeff(rij)
                    x = [3 * i, 3 * i + 1, 3 * i + 2]
                    y = [3 * j, 3 * j + 1, 3 * j + 2]
                    row.extend(x + x)
                    col.extend(x + y)
                    data.extend(3 * [-coeff] + 3 * [coeff])

        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        if initial_assembly:
            data.extend([self.mu * self.c_stab] * self.dim * N)
        else:
            data.extend([self.c_stab] * self.dim * N)

        # create the matrix
        start_time = time.time()
        self.P = sparse.csc_matrix(
            (data, (row, col)), shape=(self.dim * N, self.dim * N))
        logger.info('--- created CSC matrix in %s s ---' %
                    (time.time() - start_time))

        if not initial_assembly:
            if len(fixed_atoms) != 0:
                self.P.tolil()
            for i in fixed_atoms:
                self.P[i, :] = 0.0
                self.P[:, i] = 0.0
                self.P[i, i] = 1.0

        self.P = self.P.tocsr()

        # Create solver
        if self.use_pyamg and have_pyamg:
            start_time = time.time()
            self.ml = smoothed_aggregation_solver(
                self.P, B=None,
                strength=('symmetric', {'theta': 0.0}),
                smooth=(
                    'jacobi', {'filter': True, 'weighting': 'local'}),
                improve_candidates=[('block_gauss_seidel',
                                     {'sweep': 'symmetric', 'iterations': 4}),
                                    None, None, None, None, None, None, None,
                                    None, None, None, None, None, None, None],
                aggregate='standard',
                presmoother=('block_gauss_seidel',
                             {'sweep': 'symmetric', 'iterations': 1}),
                postsmoother=('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 1}),
                max_levels=15,
                max_coarse=300,
                coarse_solver='pinv')
            logger.info('--- multi grid solver created in %s s ---' %
                        (time.time() - start_time))

        return self.P

class Hess():

    def __init__(self, full = True, percent = 100, seed=1234):
#        self.atoms = atoms
        self.H0 = None
        self.full = full
        self.percent = percent
        self.seed = seed
    def make_precon(self, atoms, delta=0.0005, mass_weighted=False):
        """
        Calculate (mass weighted) hessian using central diff formula.
        :param atoms: atoms object with defined calculator
        :param delta: step size for numeric differentiation
        :type atoms: ase.Atoms
        :type delta: float
        :return: numpy square symmetric array
        """
        # convert delta to Angs
        delta *= Bohr
        # allocate matrix
        l = len(atoms)
        H = np.zeros((3 * l, 3 * l), dtype=np.float64)
        r = 0
        # gradients matrix
        for i, j in product(range(l), range(3)):
            g = np.zeros((l, 3))
            for k in (-1, 1):
                atoms1 = atoms.copy()
                atoms1[i].position[j] += k * delta
                atoms1.set_calculator(atoms.get_calculator())
                g += - k * atoms1.get_forces()

            H[r] = 0.5 * g.flatten()
            r += 1
        # check symmetry assuming gradients computed with 10^-3 Hartree/Bohr precision
        gprec = 0.001 * Hartree
        assert np.max(np.abs(H - H.T)) < gprec, np.max(np.abs(H - H.T))
        # Hessian
        H /= delta
        # symmetrize
        H = 0.5 * (H + H.T)
        # mass weight
        if mass_weighted:
            v = np.sqrt(atoms.get_masses()).repeat(3).reshape(-1, 1)
            H /= np.dot(v, v.T)

        
        if self.full == True:
            self.P = sparse.csr_matrix(H)
        elif self.full == False: #and self.percent != 100:
            np.random.seed(seed=self.seed)
            x = 0
            for aa in H:
                indices = np.random.choice(np.arange(aa.size), replace=False, size=int(aa.size*(1-(self.percent/100.))))
                H[x][indices] = 0
                x+=1
            self.P = sparse.csr_matrix(H)
        else:
            Hdiag = np.diag(np.diag(H))
            self.P = sparse.csr_matrix(Hdiag)
        
        
        print(self.P)
        return self.P

    def dot(self, x, y):
        return longsum(self.P.dot(x) * y)

    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        start_time = time.time()
        #if self.use_pyamg and have_pyamg:
            #y = self.ml.solve(x, x0=rand(self.P.shape[0]),
                              #tol=self.solve_tol,
                              #accel='cg',
                              #maxiter=300,
                              #cycle='W')
        #else:
        y = spsolve(self.P, x)
#        y = sparse.linalg.lsqr(self.P, x)
#        y = self.P.dot(x)
        logger.info('--- Precon applied in %s seconds ---',
                    time.time() - start_time)
        return y

class G1_pyamff(Precon):

    def __init__(self, r_cut=None, r_NN=None,
                 mu=None, mu_c=None,
                 dim=3, c_stab=0.1, force_stab=False,
                 recalc_mu=False, array_convention='C',
                 use_pyamg=True, solve_tol=1e-8,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False):

        self.r_NN = r_NN
        self.r_cut = r_cut
        self.mu = mu
        self.mu_c = mu_c
        self.estimate_mu_eigmode = estimate_mu_eigmode
        self.c_stab = c_stab
        self.force_stab = force_stab
        self.array_convention = array_convention
        self.recalc_mu = recalc_mu
        self.P = None
        self.old_positions = None

        if use_pyamg and not have_pyamg:
            use_pyamg = False
            logger.warning('use_pyamg=True but PyAMG cannot be imported! '
                           'falling back on direct inversion of '
                           'preconditioner, may be slow for large systems')

        self.use_pyamg = use_pyamg
        self.solve_tol = solve_tol
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell

        if dim < 1:
            raise ValueError('Dimension must be at least 1')
        self.dim = dim

        if not have_matscipy:
            logger.info('Unable to import Matscipy. Neighbour list '
                        'calculations may be very slow.')

#    def make_precon(self, atoms, recalc_mu=None):

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned. Note that this function will use self.mu, whatever it is.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        config = ConfigClass()
        config.initialize()

        nl = NeighborLists(cutoff=5.847888733346419)
        images=OrderedDict()
        images[0] = atoms
        nl.calculate(images,fortran=False)
        fp_paras = config.config['fp_paras'].fp_paras
        fps, fpprimes = loop_FPs_precon(0,nl, fp_paras, fortran=False)
        N=len(atoms)
        p = np.zeros((N*3, N*3))
        
        #for center in fps:
        #    for neighbor in fps[center][0]:
        #        p[center][neighbor] = [fps[center][0][neighbor]

        for key, value in fpprimes:
            p[key][value] = fpprimes[key,value][0]

        Psparse = sparse.csr_matrix(p)
#        print(Psparse)
#        print(self.mu)
#        print('lenPsparse', len(Psparse.nonzero()[0]))
        diag = Psparse.diagonal()
        diag += self.mu * self.c_stab
        Psparse.setdiag(diag)
#        Psparse.eliminate_zeros()
#        P = Psparse.data #Psparse[Psparse.nonzero()]
#        P = p.flatten()
        self.P = Psparse
        print(self.mu)
        print(self.P)
        return self.P


    def dot(self, x, y):
        """
        Return the preconditioned dot product <P x, y>

        Uses 128-bit floating point math for vector dot products
        """
        return longsum(self.P.dot(x) * y)

    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        start_time = time.time()
        if self.use_pyamg and have_pyamg:
            y = self.ml.solve(x, x0=rand(self.P.shape[0]),
                              tol=self.solve_tol,
                              accel='cg',
                              maxiter=300,
                              cycle='W')
        else:
            y = spsolve(self.P, x)
        logger.info('--- Precon applied in %s seconds ---',
                    time.time() - start_time)
        return y




