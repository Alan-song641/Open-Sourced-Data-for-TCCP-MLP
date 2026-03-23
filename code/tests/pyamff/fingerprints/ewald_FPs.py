from math import log, pi, sqrt
import sys
from scipy import constants
from scipy.special import erfc
from collections import Counter
import time

import numpy as np
import torch

from pyamff.fingerprints.ewald_lattice import Lattice

def timing_decorator(func):
    """
    Decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper


# from pymatgen
class Ewald_FPs():
    """
    Calculates the electrostatic energy of a periodic array of charges using
    the Ewald technique.


    Ref:
    Ewald summation techniques in perspective: a survey
    Abdulnour Y. Toukmaji and John A. Board Jr.
    DOI: 10.1016/0010-4655(96)00016-1
    URL: http://www.ee.duke.edu/~ayt/ewaldpaper/ewaldpaper.html

    This matrix can be used to do fast calculations of Ewald sums after species
    removal.

    E = E_recip + E_real + E_point

    Atomic units used in the code, then converted to eV.
    """

    def __init__(self,
                 structure,
                 real_space_cut=None,
                 recip_space_cut=None,
                 eta=None,
                 acc_factor=12,
                 w=1 / 2**0.5,
                 compute_forces=True,
                 chg_grad=True, # whether compute charge gradient for cell_energy
                 chg_offset=0.0,
                 total_ion_charge=None, # only work when the system has counter F ions!!!
                 if_force=True,
                 ):
        """
        Initializes and calculates the Ewald sum. Default convergence
        parameters have been specified, but you can override them if you wish.

        Args:
            structure (ase.Atom): Originally the type is pymatgen.core.structure 
                but here we use ase.Atom().
            real_space_cut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum. Defaults to None,
                which means determine automagically using the formula given
                in gulp 3.1 documentation.
            recip_space_cut (float): Reciprocal space cutoff radius.
                Defaults to None, which means determine automagically using
                the formula given in gulp 3.1 documentation.
            eta (float): The screening parameter. Defaults to None, which means
                determine automatically.
            acc_factor (float): No. of significant figures each sum is
                converged to.
            w (float): Weight parameter, w, has been included that represents
                the relative computational expense of calculating a term in
                real and reciprocal space. Default of 0.7 reproduces result
                similar to GULP 4.2. This has little effect on the total
                energy, but may influence speed of computation in large
                systems. Note that this parameter is used only when the
                cutoffs are set to None.
            compute_forces (bool): Whether to compute forces. False by
                default since it is usually not needed.
        """

        # Converts unit of q*q/r into eV
        self.CONV_FACT = 1e10 * constants.e / (4 * pi * constants.epsilon_0)

        self._lattice = Lattice(structure.get_cell()[:])

        # self._charged = abs(structure.charge) > 1e-8  # bool
        self._vol = structure.get_volume()
        self._compute_forces = compute_forces

        self._acc_factor = acc_factor

        # set screening length
        self._eta = eta or (len(structure) * w / (self._vol**2))**(1 / 3) * pi
        self._sqrt_eta = sqrt(self._eta) # which is alpha in the equation

        # acc factor used to automatically determine the optimal real and
        # reciprocal space cutoff radii
        self._accf = sqrt(log(10**self._acc_factor))

        '''
        NOTE: real_space_cut value:

        1. > 0: consider real term, default
        2. = 0: consider real term, automatically determine the optimal rcut and gcut
        '''
        if real_space_cut == 0.0:
            real_space_cut = None

        if real_space_cut is not None and real_space_cut > 0.0:
            self.recip_only = True
            self.real_space_cut = real_space_cut
        else:
            self.recip_only = False
        
        # rcut (influenced by atom num and volume!)
        # self._rmax * self._gmax = Constance(55.2620422318571) = 2 * self._accf * self._accf
        # self._rmax = real_space_cut or self._accf / self._sqrt_eta
        # self._gmax = recip_space_cut or 2 * self._accf * self._accf / self._rmax
        self._rmax = self._accf / self._sqrt_eta
        self._gmax = 2 * self._accf * self._accf / self._rmax

        self.if_force = if_force
        self.chg_grad = chg_grad
        '''
        # The next few lines pre-compute certain quantities and store them.
        # Ewald summation is rather expensive, and these shortcuts are
        # necessary to obtain several factors of improvement in speedup.
        '''

        # Alan: get pred_qi based on chemical symbols to determine which NN output
        self.chem_symbol = np.array(structure.get_chemical_symbols())
        self.chem_num_dict = dict(Counter(self.chem_symbol))

        self._coords = structure.get_positions()
        self._frac_coords = self._lattice.get_fractional_coords(self._coords)
        self.numsites = len(structure)

        # Define the private attributes to lazy compute reciprocal and real
        # space terms.
        self._initialized = False
        self._recip = None
        self._real, self._point = None, None
        self._forces = None

        # qi(s)
        self._oxi_states = torch.zeros(self.numsites, dtype=torch.double)
        
        # system real net charge: Q
        try:
            qtot = float(np.sum(structure.get_initial_charges())) + chg_offset
        except Exception:
            qtot = 0.0 + chg_offset

        self.total_charge = torch.tensor([qtot], dtype=torch.double, requires_grad=True)

        # initial variables for energy calculation
        Elec = self.init_for_real()
        self.init_for_recip()

        # NOTE: use Qeq method
        A_analy = self.init_for_qeq_analy(Elec) 


    def _apply_chi_qtot_linear(self, chi0: torch.Tensor) -> torch.Tensor:
        """Override per-atom electronegativities using chi_i(Q)=a+b*Q.

        The coefficients are computed from the top-left nxn block of `self.A`:
            a = (1^T A^{-1} chi0) / (1^T A^{-1} 1)
            b = 1 / (1^T A^{-1} 1)
            chi_i(Q) = a + b * Q

        Notes:
        - `chi0` must be a length-n vector (no dummy node).
        - `self.A` is expected to be (n+1)x(n+1); this uses `self.A[:-1,:-1]`.
        """

        n = self.numsites
        if chi0.numel() != n:
            raise ValueError(f"chi0 must have length {n}, got {chi0.numel()}")

        A_nn = self.A[:n, :n]
        ones = torch.ones(n, dtype=A_nn.dtype, device=A_nn.device)

        # Use solve() rather than explicit inverse for stability and autograd.
        Ainv_ones = torch.linalg.solve(A_nn, ones)
        denom = torch.dot(ones, Ainv_ones)
        Ainv_chi0 = torch.linalg.solve(A_nn, chi0)

        # NOTE: weight should also be the function of R, but here I just use the weight as a constant, which is not accurate so the dE/dR benchmark not pass!!
        weight = torch.dot(ones, Ainv_chi0) / denom 
        bias = 1.0 / denom
        # bias = torch.dot(ones, Ainv_chi0) / denom

        # chi_eff = weight * self.total_charge + bias
        chi_eff = chi0 + weight * self.total_charge 
        # print(chi_eff)

        return chi_eff

        # chi_eff = chi_eff.to(dtype=chi0.dtype, device=chi0.device)
        # return torch.ones((n,), dtype=chi0.dtype, device=chi0.device) * chi_eff

        # benchmarking test
        # A_auto = self.init_for_qeq()
        # print(torch.allclose(A_analy, A_auto))
        # print(A_auto - A_analy)



    def set_oxi_states(self, X_dict, J_dict, start_index, chi_qtot_linear=False):
        """
        Alan: the real implementation of Qeq method, by solving N+1 linear equations
        From self._ele_negativity to self._oxi_states
       
        NOTE: Call this function in NN forward function 
        NOTE: Do not loose the require_grad attribute 
        """
        
        index_dict = {}

        # Update the total charge (maintain the require_grad attribute)
        self._ele_negativity[-1] = -self.total_charge
        
        # Count element in THIS IMAGE
        for element in start_index.keys():
            index_dict[element] = 0

        for element, i in zip(self.chem_symbol, range(len(self.chem_symbol))):
            self._ele_negativity[i] = X_dict[element][start_index[element] + index_dict[element]]
            self._hardness[i] = J_dict[element][start_index[element] + index_dict[element]]
            index_dict[element] += 1

        # update A with +J
        diag_J = torch.diag_embed(self._hardness)
        self.A = torch.add(self.A, diag_J)

        # Optional: override electronegativities using chi_i(Q)=a+b*Q.
        if chi_qtot_linear:
            chi0 = self._ele_negativity[:-1]
            self._ele_negativity[:-1] = self._apply_chi_qtot_linear(chi0)

        # qi(s) plus lambda
        q_prime = torch.linalg.solve(self.A, -self._ele_negativity.view(-1, 1))
        # print(torch.allclose(self.A @ q_prime, self._ele_negativity.view(-1,1)))

        # NOTE:BENCHMARK: /sum(Aij*qi) + Xi + λ = 0
        # print(torch.sum(self.A[0:-1, 0:-1] @ self._oxi_states.view(-1, 1) + self._ele_negativity[0:-1].view(-1, 1) + self._lambda))

        # NOTE: the last value is lambda 
        self._oxi_states = q_prime[0: -1].flatten()
        self._lambda = q_prime.data[-1].flatten()
        return self._oxi_states, self._lambda, self._ele_negativity[0:-1], self._hardness[0:-1]

    

    # #@timing_decorator
    def init_for_qeq(self):

        """
        Alan: the real implementation of Qeq method, by solving N+1 linear equations
        NOTE: build the A', q' matrix (the b matrix is built in set_oxi_states function)
        From self._ele_negativity to self._oxi_states
       
        NOTE: Call this function in NN forward function 
        NOTE: New implementation with torch.autograd function
        """

        # Xi(s) plus qtot with size of N+1*1
        self._ele_negativity = torch.zeros(self.numsites + 1, dtype=torch.double)
        # self._ele_negativity[-1] = -self.total_charge # will revert in set_oxi_states()

        self._hardness = torch.zeros(self.numsites + 1, dtype=torch.double)

        # NOTE: pseudo charge info to calculate A (for benchmarking)
        _ele_chg = {'O': 5.6, 'Ti': 3.1, 'H': 6.2, 'C': 4.7, 'F': 6.1, 'S': 3.8, 'V': 3.2, 'W': 3.6, 'Ta': 3.1, 'Si': 3.3, 'Ni': 3.3, 'Au': 3.6, 'Ge': 3.3, 'Pd': 4.1, 'Pt': 3.5,'Cu': 3.8, 'Al': 3.0, 'Fe': 3.2, 'Mn': 3.1, 'Co': 3.2, 'Cr': 3.1, 'Mo': 3.4, 'Nb': 3.2, 'Zr': 3.4}
        self._oxi_states = torch.tensor([_ele_chg[ele] for ele in self.chem_symbol], requires_grad=True)

        # qi(s) plus lambda
        self.A = torch.zeros((self.numsites + 1, self.numsites + 1), dtype=torch.double)
        self.A[:, self.numsites] = 1
        self.A[self.numsites, :] = 1
        self.A[self.numsites, self.numsites] = 0 # system whole charge

        Utot = self.total_energy / self.CONV_FACT # NOTE: the output charge unit should maintain as e(electron), thus Utot not need to convert to unit eV

        # dEqeq/dq = 0
        first_deriv = torch.autograd.grad(Utot, self._oxi_states, retain_graph=True, create_graph=True)[0]
        for i, a in enumerate(first_deriv):
            self.A[i][0: self.numsites] = torch.autograd.grad(a, self._oxi_states, retain_graph=True, create_graph=True)[0]
        
        # NOTE: the hardness
        # diag_J = torch.diag_embed(torch.tensor([hardness_dict[chem] for chem in self.chem_symbol]))
        # self.A[0:self.numsites, 0:self.numsites] += diag_J

        return self.A.clone().detach()


    #@timing_decorator
    def init_for_qeq_analy(self, Elec):
        """
        Alan: the real implementation of Qeq method, by solving N+1 linear equations
        NOTE: build the A', q' matrix (the b matrix is built in set_oxi_states function)
        From self._ele_negativity to self._oxi_states
       
        NOTE: Call this function in NN forward function 
        NOTE: Do not loose the require_grad attribute 
        """

        # qi(s) plus lambda
        # self._oxi_states = torch.zeros(self.numsites, requires_grad=True, dtype=torch.double)

        # Xi(s) plus qtot with size of N+1*1
        self._ele_negativity = torch.zeros(self.numsites + 1, dtype=torch.double)
        # self._ele_negativity[-1] = -self.total_charge # will revert in set_oxi_states()

        # atomic hardness, length = N+1, the last one is the dummy node with value zero (ensure the same size as self.A)
        self._hardness = torch.zeros(self.numsites + 1, dtype=torch.double)

        # matrix A with size of N+1*N+1
        self.A = torch.zeros((self.numsites + 1, self.numsites + 1), dtype=torch.double)
        self.A[:, self.numsites] = 1
        self.A[self.numsites, :] = 1
        self.A[self.numsites, self.numsites] = 0 # system whole charge

        # NOTE: the reciprocal term
        Elec += self.e_recip * self.prefactor * 2**0.5

        # NOTE: the self term
        Elec -= torch.eye(self.numsites) * sqrt(self._eta / pi) 

        # NOTE: if include the cell energy derivative in the A matrix
        if self.chg_grad:
            Elec -= pi / self._vol / self._eta / 2

        # NOTE: now turns to diagonal component (i=j, self term, and 1/2*Jii*qi**2), multiply the elements in the diagonal with factor of 2
        #       for non-diagonal component(i!=j, reciprocal, real term with eij+eji), symmetry so it is same as *= 2
        self.A[0:self.numsites, 0:self.numsites] = Elec + Elec.T

        # self.A *= self.CONV_FACT # NOTE: the output charge unit should maintain as e(electron)

        # NOTE: NOW I want the J(atomic hardness) also the function of g
        # diag_J = torch.diag_embed(torch.tensor([hardness_dict[chem] for chem in self.chem_symbol]))
        # self.A[0:self.numsites, 0:self.numsites] += diag_J

        return self.A 
    
    def _calc_qeq_term(self, q_grad=False):
        
        if q_grad:
            _oxi = self._oxi_states
        else:
            _oxi = self._oxi_states.clone().detach()

        _ele = self._ele_negativity[:-1]
        _hard = self._hardness[:-1]

        # NOTE: sum_i_N{Xi*qi+0.5*Ji*qi^2} 
        # self._qeq_term = torch.zeros(self.numsites, dtype=torch.double)
        # for i, chem in enumerate(self.chem_symbol):
        #     self._qeq_term[i] = self._ele_negativity[i] * _oxi[i]  + 0.5 * self._hardness[i] * _oxi[i]**2
        #     self._qeq_term[i] *= self.CONV_FACT

        # do we need this CONV_FACT here?
        self._qeq_term = self.CONV_FACT*(_ele * _oxi  + 0.5 * _hard * torch.square(_oxi))
        # self._qeq_term = _ele * _oxi  + 0.5 * _hard * torch.square(_oxi)

        return torch.sum(self._qeq_term)


    #@timing_decorator
    def init_for_recip(self):
        """
        Alan: precalculate the reciprocal variables instead in the forward function
        (self.erecip):                                   
        """
        self.prefactor = 2 * pi / self._vol

        # alike real space but h,k,l = 2pi/Lx, 2pi/Ly, 2pi/Lz
        rcp_latt = self._lattice.reciprocal_lattice

        # neighbor_list = [(coord, distance, int(index), image)]
        # same treatment as real space, but xyz become hkl defining the reciprocal space
        # NOTE: is ri not rij
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], self._gmax)
                                                    
        # frac_coords of the neighbors NOT self._frac_coords
        frac_coords = [
            fcoords for (fcoords, dist, i, img) in recip_nn if dist != 0
        ]

        self.gs = torch.from_numpy(rcp_latt.get_cartesian_coords(frac_coords))

        # norm of the hkl: G*G = k^2 (k is discrete!)
        self.g2s = torch.sum(self.gs**2, 1)

        self.expvals = torch.exp(-self.g2s / (4 * self._eta))

        # k*ri = (910 X 1 X 3) * (1 X 96 X 3) = (910 X 96 X 3)
        # grs = (910 X 96)
        grs = torch.sum(self.gs[:, None] * self._coords[None, :], 2)

        self.e_recip = torch.zeros((self.numsites, self.numsites), dtype=torch.double)
                                  
        # if self._compute_forces:
        #     self.forces = torch.zeros((self.numsites, 3), dtype=torch.double)
        '''Require qi and qj so can not put in recip_init'''
        # oxistates = self._oxi_states

        # # calculate the structure factor
        # if self._compute_forces:
        #     sreals = torch.sum(oxistates[None, :] * torch.cos(grs), 1)
        #     simags = torch.sum(oxistates[None, :] * torch.sin(grs), 1)
        # else:
        # sreals = range(len(self.gs))
        # simags = range(len(self.gs))

        for g2, gr, expval in zip(self.g2s, grs, self.expvals):

            # k is constant in every loop, eg: k = [0, 0, 1], [0, 0, 2] ....
            # gr[None, :] = (1 X 96)
            # gr[:, None] = (96 X 1)
            # m = k*ri - k*rj = (96 X 96)
            m = (gr[None, :] + pi / 4) - gr[:, None] 
            m = torch.sin(m)  # equal to m = np.sin(m)
            m *= expval / g2

            self.e_recip += m
    
    
    #@timing_decorator
    def _calc_recip(self):
        """
        Perform the reciprocal space summation. Calculates the quantity
        E_recip = 1/(2PiV) sum_{G < Gmax} exp(-(G.G/4/eta))/(G.G) S(G)S(-G)
        where
        S(G) = sum_{k=1,N} q_k exp(-i G.r_k)
        S(G)S(-G) = |S(G)|**2.

        This method is heavily vectorized 
        
        NOTE: call this func in the forward func in NN
        """
        # # numsites = self._s.num_sites
        # prefactor = 2 * pi / self._vol

        # erecip = torch.zeros((self.numsites, self.numsites), dtype=torch.double)
        forces = torch.zeros((self.numsites, 3), dtype=torch.double)

        # coords = self._coords

        # # alike real space but h,k,l = 2pi/Lx, 2pi/Ly, 2pi/Lz
        # rcp_latt = self._lattice.reciprocal_lattice

        # # neighbor_list = [(coord, distance, int(index), image)]
        # # same treatment as real space, but xyz become hkl defining the reciprocal space
        # recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
        #                                          self._gmax)

        # frac_coords = [
        #     fcoords for (fcoords, dist, i, img) in recip_nn if dist != 0
        # ]

        # # hkl
        # gs = rcp_latt.get_cartesian_coords(frac_coords)

        # # norm of the hkl: G.G = k^2 (k is discrete!)
        # g2s = torch.sum(gs**2, 1)

        # expvals = torch.exp(-g2s / (4 * self._eta))

        # # k*r
        # grs = torch.sum(gs[:, None] * coords[None, :], 2)

        # create array where q_2[i,j] is qi * qj  qiqj should not requre_grad!!!
        '''
        NOTE:
        # contains require_grad for Energy --> dE/dq|r * dq/dr
        # not contains require_grad for Forces dE/dr|q * dr/dr
        '''
        qiqj = self._oxi_states[None, :] * self._oxi_states[:, None]
        self.e_recip *= self.prefactor * self.CONV_FACT * qiqj * 2**0.5
        
        if self.if_force:
            oxistates = self._oxi_states

            # repeat the grs because memory is too big in init_recip
            grs = torch.sum(self.gs[:, None] * self._coords[None, :], 2)
            cosgrs = torch.cos(grs)
            singrs = torch.sin(grs)

            # calculate the structure factor
            sreals = torch.sum(oxistates[None, :] * cosgrs, 1)
            simags = torch.sum(oxistates[None, :] * singrs, 1)

            for g, g2, cosgr, singr, expval, sreal, simag in zip(self.gs, self.g2s, cosgrs, singrs, self.expvals, sreals,simags):
                                                    
                # Uses the identity sin(x)+cos(x) = (2**0.5) * sin(x + pi/4)
                # m = (gr[None, :] + pi / 4) - gr[:, None]
                # m = torch.sin(m)
                # m *= expval / g2

                # erecip += m

                pref = 2 * expval / g2 * oxistates
                factor = self.prefactor * pref * (sreal * singr - simag * cosgr)
                                                    
                forces += factor[:, None] * g[None, :]

            forces *= self.CONV_FACT
        else:
            forces = 0
        return self.e_recip, forces
    
    
    #@timing_decorator
    def init_for_real(self):
        """
        Alan: precalculate the real and point variables instead in the forward function
        (nfcoords, rij, js):                     
        """

        # self.ereal = torch.zeros((self.numsites, self.numsites), dtype=torch.double)
        self.e_real = torch.tensor(0.0, dtype=torch.double)

        # the term of (1 - fsc)(qi*qj/rij)
        self.e_screen = torch.zeros(self.numsites, dtype=torch.double)

        # len: self.numsites
        self.rxyzs = []
        self.rij_list = []
        self.js_list = []
        
        # matrix form of electrostatic energy = q^T*Elec*q
        Elec = torch.zeros((self.numsites, self.numsites), dtype=torch.double)

        for i in range(self.numsites):
            # rij: distance between center i and js (within the _rmax)
            # js: ID of the atom (alike neighs_incell)
            nfcoords, rij, js, _ = self._lattice.get_points_in_sphere(
            self._frac_coords,
            self._coords[i],
            self._rmax,
            zip_results=False,
            )
            
            if self.recip_only:
                inds = rij > self.real_space_cut
            else:
                # remove the rii term
                inds = rij > 1e-8

            js = torch.from_numpy(js[inds])
            rij = torch.from_numpy(rij[inds])
            nfcoords = torch.from_numpy(nfcoords[inds])
            nccoords = self._lattice.get_cartesian_coords(nfcoords)

            if self.if_force:
                rxyz = torch.tensor([self._coords[i]]) - nccoords
            else:
                rxyz = 0.0

            self.rxyzs.append(rxyz)

            self.rij_list.append(rij)
            self.js_list.append(js)

            # Precompute erfc and division to avoid redundant computation
            erfcval = erfc(self._sqrt_eta * rij) / rij

            # Use torch.scatter_add for efficient accumulation
            Elec[i].scatter_add_(0, js, erfcval * 0.5)
        
        return Elec

    #@timing_decorator
    def _calc_real_and_point(self):
        """
        Determines the self energy -(eta/pi)**(1/2) * sum_{i=1}^{N} q_i**2

        NOTE: call this func in the forward func in NN
        """

        forces = torch.zeros((self.numsites, 3), dtype=torch.double)

        forcepf = 2.0 * self._sqrt_eta / sqrt(pi)

        '''
        NOTE:
        # contains require_grad for Energy --> dE/dq|r * dq/dr
        # not contains require_grad for Forces dE/dr|q * dr/dr
        '''
        qs = self._oxi_states 

        # original version of epoint
        self.e_point = -(qs**2) * sqrt(self._eta / pi)

        # not screening (less looped for speed acceleration)
        for rij, rxyz, js, i in zip(self.rij_list, self.rxyzs, self.js_list, range(self.numsites)):
            '''
            # rij: distance between center atom i and neighbors js (within the _rmax)
            # js: IDs of the neighbor of center atom i  (alike neighs_incell is a list!)
            '''

            # nfcoords, rij, js, _ = self._lattice.get_points_in_sphere(
            #     self._frac_coords,
            #     self._coords[i],
            #     self._rmax,
            #     zip_results=False)

            # # remove the rii term (but include i and i_ghost)
            # inds = rij > 1e-8
            # js = js[inds]
            # rij = rij[inds]
            # nfcoords = nfcoords[inds]

            qi = qs[i]
            qj = qs[js]

            # center functions!!
            val = qi * qj / rij

            erfcval = erfc(self._sqrt_eta * rij)
            new_ereals = erfcval * val

            self.e_real += torch.sum(new_ereals) # not need to make e_real a matrix
            # print(self.ereal[i])

            # after screening: E_ab_sc = E_ab_tot - (1 - fsc)E_ab_short
            # self.e_screen[i] = 0.0 # deprecated
            if self.if_force:
                # NOTE: force first term dE/dr|q
                qi_ = qi
                qj_ = qj

                fijpf = qj_ / rij**3 * (erfcval + forcepf * rij * torch.exp(-self._eta * rij**2))
                
                # the sequence of force is the same as self._coords[i] 
                # which is the same as ase.Atoms.get_positions()
                forces[i] += torch.sum(torch.unsqueeze(fijpf, 1) * rxyz * qi_ * self.CONV_FACT, axis=0,)
            else:
                forces = 0
            
        self.e_real *= 0.5 * self.CONV_FACT        
        # self.e_screen *= 0.5 * self.CONV_FACT
        self.e_point *= self.CONV_FACT

        if self.recip_only:
            self.e_point *= 0
        return self.e_real, self.e_point, forces
    
    #@timing_decorator
    def _cal_cell_energy(self):
        '''extra energy if the cell is not charge neutral'''
   
        # NOTE: if include the cell energy derivative in the A matrix
        if self.chg_grad:
            total_charge = torch.sum(self._oxi_states)
        else:
            total_charge = self.total_charge
        return -self.CONV_FACT / 2 * pi / self._vol / self._eta * total_charge**2

    def _calc_ewald_terms(self):
        """Calculates and sets all Ewald terms (point, real and reciprocal)."""
        # get additional charges if the system is not charge neutral
        
        _recip, self.f_recip = self._calc_recip()
        _real, e_point, self.f_real = self._calc_real_and_point()
        self._charged_cell_energy = self._cal_cell_energy()
        # if self._compute_forces:
        #     self.recip_forces = recip_forces 
        #     self.real_point_forces = real_point_forces


    @property
    def reciprocal_space_energy(self):
        """The reciprocal space energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self.e_recip


    @property
    def real_space_energy(self):
        """The real space energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self.e_real


    @property
    def point_energy(self):
        """The point energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self.e_point


    @property
    def total_energy(self):
        """The total energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True

        self.e_recip = torch.sum(self.e_recip)
        # self.e_real = torch.sum(self.e_real)
        self.e_point = torch.sum(self.e_point)

        return self.e_recip + self.e_real + self.e_point + self._charged_cell_energy
        # return self.e_recip + self.e_real + self.e_point
        # return self._charged_cell_energy
    
        # def set_net_charge_energy(self, ifequil_charge=True):
    #     """
    #     Alan: Compute the correction for a charged cell
    #     net predicted charge of the system: /sum{qj} from j to N

    #     NOTE: DEPRECATED with qeq
    #     NOTE: Call this function in NN forward function 
    #     NOTE: Do not loose the require_grad attribute
    #     """
    #     if ifequil_charge:
            
    #         # NOTE: self.total_charge  NO GRAD!
    #         # corrected partial charges
    #         self._oxi_states -= (torch.sum(self._oxi_states) - self.total_charge) / self.numsites

    #     else:
    #         self.total_charge = torch.sum(self._oxi_states) # NOTE: YES GRAD!
        
    #     # get additional charges if the system is not charge neutral
    #     self._charged_cell_energy = torch.tensor(-self.CONV_FACT / 2 * pi /
    #                                 self._vol / self._eta *
    #                                 self.total_charge**2)
        
    #     return self._oxi_states


    # def set_oxi_states(self, charge_dict, start_index):
    #     """
    #     Alan: get pred_qi based on chemical symbols to determine which NN output
       
    #     NOTE: Call this function in NN forward function 
    #     NOTE: Do not loose the require_grad attribute
    #     """
    #     index_dict = {}

    #     # set charges in THIS IMAGE
    #     for element in start_index.keys():
    #         index_dict[element] = 0

    #     for element, i in zip(self.chem_symbol, range(len(self.chem_symbol))):
    #         self._oxi_states[i] = charge_dict[element][start_index[element] + index_dict[element]]
    #         index_dict[element] += 1
        
    #     return