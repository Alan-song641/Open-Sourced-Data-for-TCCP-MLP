import numpy as np
import pickle
import math
import os
import time
import torch
from scipy import sparse
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
from pyamff.utilities import fileIO as io
from pyamff.utilities.preprocessor import normalizeParas
from collections import OrderedDict
import itertools
import tempfile
import bz2file as bz2
try:
    from pyamff.fmodules import fbp
    FMODULES = True
#    print("Fortran BP fingerprints loaded")
except:
    FMODULES = False
#    print("Fortran BP fingerprints not loaded")


class BehlerParrinello():

    """
    An implementation of the Behler-Parrinello descriptors.

    References
    ----------
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401.

    """

    def __init__(self, r_cut=6.0, r_s=0., eta=1., lambda_=1., zeta=1.):
        self.r_cut = r_cut
        self.r_s = r_s
        self.eta = eta
        self.zeta = zeta
        self.lambda_ = lambda_
        self._system_elements = None
        self._elements = None
        self._element_pairs = None
        self._N_unitcell = None
        self.type = None

    # Lei added to set parameters
    def set(self, g):
        self.type = g.subtype
        #print('gg', g.__dict__)
        if self.type == 'G1':
            self.r_s = g.Rs
            self.eta = g.eta
            self.neighbor = g.neighbor
            self.center = g.center
        if self.type == 'G2':
            self.r_s = g.thetas
            self.eta = g.eta
            self.zeta = g.zeta
            self.lambda_ = g.lambda_
            self.neighbor1 = g.neighbor1
            self.neighbor2 = g.neighbor2
            self.center = g.center

    def get_fp_precon(self, symbol, neighborindices, neighborsymbols, Rs, G_paras, angles_list, xyzs, atom, fortran, l):

        num_Gs = len(G_paras[symbol])
        fingerprint = [None] * num_Gs
        f_neighborsymbols = [symb.ljust(2) for symb in neighborsymbols]
        for count in range(num_Gs):
            g = G_paras[symbol][count]
            self.set(g)
            if g.subtype == 'G1':
                if fortran and FMODULES:
                    value = fbp.fg1(self.neighbor, neighborindices, f_neighborsymbols, Rs, self.eta, self.r_s, self.r_cut)
                else:
                    value = self.calculate_G1_precon(neighborindices, neighborsymbols, Rs, atom, l)
            elif g.subtype == 'G2':
                if fortran and FMODULES:
                    value = fbp.fg2(self.neighbor1, self.neighbor2, neighborindices, f_neighborsymbols, Rs, self.eta, \
                                    self.r_s, self.r_cut, angles_list, np.transpose(xyzs), self.lambda_, self.zeta)
                else:
                    value = self.calculate_G2(neighborindices, neighborsymbols, Rs, angles_list, xyzs)
            fingerprint[count] = value
        #return symbol, fingerprint
        return fingerprint

    def calculate_G1_precon(self, neighborindices, neighborsymbols, Rs, atom, l):

        values = {}
        sum_val = 0.
        for count in range(len(neighborindices)):
            neigh = neighborsymbols[count]
            Rij = Rs[count]
            if neigh != self.neighbor:
                continue
            #values = np.exp(-self.eta * ((Rij - self.r_s)**2)/(self.r_cut**2)) * (0.5* (np.cos(np.pi * Rij / self.r_cut) + 1))

            values[(3*neighborindices[count])+l] = np.exp(-self.eta * ((Rij - self.r_s)**2)/(self.r_cut**2)) * (0.5* (np.cos(np.pi * Rij / self.r_cut) + 1))
            sum_val += np.exp(-self.eta * ((Rij - self.r_s)**2)/(self.r_cut**2)) * (0.5* (np.cos(np.pi * Rij / self.r_cut) + 1))
            #que = np.bincount(atom, -values[neighborindices[count]+atom], minlength=256)i

        values[(3*atom)+l] = sum_val
        #print(values)
        return values
            #yield values

    def get_fp_primes_precon(self, index, symbol, neighborindices, neighborsymbols, Rs, unitVects, m, l, G_paras, angles_list, vects, xyzs, fortran):

        num_Gs = len(G_paras[symbol])
        fingerprintprime = [None] * num_Gs
        f_neighborsymbols = [symb.ljust(2) for symb in neighborsymbols]
        for count in range(num_Gs):
            g = G_paras[symbol][count]
            self.set(g)
            if g.subtype == 'G1':
                if fortran and FMODULES:
                    f_unitVects = np.array(unitVects)
                    value = fbp.fdg1(self.neighbor, neighborindices, f_neighborsymbols, Rs, \
                                     np.transpose(f_unitVects), index, m, l, self.eta, self.r_s, self.r_cut)
                else:
                    value = self.calculate_dG1_precon(neighborindices, f_neighborsymbols, Rs, \
                                                      unitVects, index, m, l)
            elif g.subtype == 'G2':
                if fortran and FMODULES:
                    f_vects = np.array(vects)
                    value = fbp.fdg2(self.neighbor1, self.neighbor2, neighborindices, f_neighborsymbols, \
                    Rs, np.transpose(f_unitVects), index, m, l, angles_list, np.transpose(f_vects), \
                    np.transpose(xyzs), self.eta, self.r_s, self.r_cut, self.lambda_, self.zeta)
                else:
                    value = self.calculate_dG2(neighborindices, neighborsymbols, Rs, unitVects, index, m, l, angles_list, vects, xyzs)
            fingerprintprime[count] = value
        return fingerprintprime


    def calculate_dG1_precon(self, neighborindices, neighborsymbols, Rs, unitVects, i, m, l):

        values = 0.

        for count in range(len(neighborindices)):
                neighborsymbol = neighborsymbols[count]
                j = neighborindices[count]
                Rij = Rs[count]
                Rij_uv = unitVects[count]
                if neighborsymbol == self.neighbor:
                    dRijdRml = dRij_dRml(i, j, Rij_uv, m, l)
                    if dRijdRml != 0:
                        values += ((dRijdRml * (-2*self.eta * (Rij - self.r_s) *  (0.5* (np.cos(np.pi * Rij / self.r_cut) + 1))/(self.r_cut**2)) + (0.5 * (-np.pi / self.r_cut) * np.sin(np.pi * Rij / self.r_cut) * dRijdRml )) * np.exp(-self.eta * ((Rij - self.r_s) ** 2) /(self.r_cut**2) ))

        return values


    def get_fp(self, symbol, neighborindices, neighborsymbols, Rs, G_paras, angles_list, xyzs, fortran):

        """ 
        BehlorParrinello.get_fp()
        =========================
        Calculates fingerprints of atom

        Parameters
        ----------
        symbol : Atomic symbol of atom
        neighborindices : list of all neighbor indices for atom, calculated by NeighborList
        neighborsymbols : list of neighbor element types
        Rs : Rij values for each atom-neighbor pair
        G_paras : Fingerprint parameters
        angles_list : list of angles for all jik pairs
        xyzs : coordinates of neighboratoms to calculate jk distances
        fortran : flag to use fortran module
        """

        num_Gs = len(G_paras[symbol])
        fingerprint = [None] * num_Gs
        #Convert every neighbor symbols into a2 format (e.g. 'H' -> 'H ')
        f_neighborsymbols = [symb.ljust(2) for symb in neighborsymbols]
        for count in range(num_Gs):
            g = G_paras[symbol][count]
            self.set(g)
            if g.subtype == 'G1':
                if fortran and FMODULES:
                    #print ("fortran BP")
                    value = fbp.fg1(self.neighbor,neighborindices,f_neighborsymbols,Rs,self.eta,self.r_s,self.r_cut)
                else:
                    value = self.calculate_G1(neighborindices, neighborsymbols, Rs)
            elif g.subtype == 'G2':
                if fortran and FMODULES:
                    value = fbp.fg2(self.neighbor1,self.neighbor2,neighborindices,f_neighborsymbols, Rs, self.eta, \
                    self.r_s, self.r_cut, angles_list, np.transpose(xyzs), self.lambda_, self.zeta)
                else:
                    value = self.calculate_G2(neighborindices, neighborsymbols, Rs, angles_list, xyzs)
            fingerprint[count] = value
        return symbol, fingerprint


    def calculate_G1(self, neighborindices, neighborsymbols, Rs):
        """
        BehlorParrinello.calculate_G1()
        ===============================
        Function to calculate G1s for each atom

        Parameters
        ----------
        neighborindices : list of all neighbor indices for atom, calculated by NeighborList
        neighborsymbols : list of neighbor element types
        Rs : Rij values for each atom-neighbor pair
        """

        values = 0.
        for count in range(len(neighborindices)):
            neigh = neighborsymbols[count]
            Rij = Rs[count]
            if neigh != self.neighbor:
                continue
            values += np.exp(-self.eta * ((Rij - self.r_s)**2)/(self.r_cut**2)) * (0.5* (np.cos(np.pi * Rij / self.r_cut) + 1)) 
        return values


    def calculate_G2(self, neighborindices, neighborsymbols, Rs, angles_list, xyzs):
        """
        BehlorParrinello.calculate_G2()
        ===============================
        Function to calculate G2s for each atom

        Parameters
        ----------
        neighborindices : list of all neighbor indices for atom, calculated by NeighborList
        neighborsymbols : list of neighbor element types
        Rs : Rij values for each atom-neighbor pair
        angles_list : list of angles for all jik pairs
        xyzs : coordinates of neighboratoms to calculate jk distances
        """

        values = 0.
        q = 0
        counts = range(len(neighborindices))
        for y in counts:
            for z in counts[(y+1):]:
                neigh1 = neighborsymbols[y]
                neigh2 = neighborsymbols[z]
                if neigh1 != self.neighbor1 or neigh2 != self.neighbor2:
                    q += 1
                    continue
                Rij = Rs[y]
                Rik = Rs[z]
                j = neighborindices[y]
                k = neighborindices[z]
                index_angle = angles_list[q]
                j_coord = xyzs[j]
                k_coord = xyzs[k]
                jk_vector = k_coord - j_coord
                jk_dist = np.linalg.norm(jk_vector)
                values += ((2 ** (1-self.zeta)) * ((1 + self.lambda_ * np.cos(np.radians(index_angle))) ** self.zeta) \
                          * np.exp(-self.eta * (Rij**2)/(self.r_cut**2)) \
                          * np.exp(-self.eta * (Rik**2)/(self.r_cut**2)) \
                          * np.exp(-self.eta * (jk_dist**2)/(self.r_cut**2)) \
                          * (0.5 * (np.cos(np.pi * Rij / self.r_cut) + 1)) \
                          * (0.5 * (np.cos(np.pi * Rik / self.r_cut) + 1)) \
                          * (0.5 * (np.cos(np.pi * jk_dist / self.r_cut) + 1)))
                q += 1
        return values


    def get_fp_primes(self, index, symbol, neighborindices, neighborsymbols, Rs, unitVects, m, l, G_paras, angles_list, vects, xyzs, fortran):

        """ 
        BehlorParrinello.get_fp_primes()
        =========================
        Calculates derivatives of fingerprints of atom

        Parameters
        ----------
        index : index of central atom
        symbol : Atomic symbol of atom
        neighborindices : list of all neighbor indices for atom, calculated by NeighborList
        neighborsymbols : list of neighbor element types
        Rs : Rij values for each atom-neighbor pair
        unitVects : unit vectors for ij pairs
        m : index of neighbor
        l : direction (x,y,z) = (0,1,2)
        G_paras : Fingerprint parameters
        angles_list : list of angles for all jik pairs
        vects : ij pair vectors 
        xyzs : coordinates of neighboratoms to calculate jk distances
        fortran : flag to use fortran module
        """

        num_Gs = len(G_paras[symbol])
        fingerprintprime = [None] * num_Gs
        f_neighborsymbols = [symb.ljust(2) for symb in neighborsymbols]
        for count in range(num_Gs):
            g = G_paras[symbol][count]
            self.set(g) 
            if g.subtype == 'G1':
                if fortran and FMODULES:
                    f_unitVects = np.array(unitVects)
                    value = fbp.fdg1(self.neighbor, neighborindices, f_neighborsymbols,Rs,\
                    np.transpose(f_unitVects), index, m, l, self.eta, self.r_s, self.r_cut)

                else:
                    value = self.calculate_dG1(neighborindices, neighborsymbols, Rs, \
                                                   unitVects, index, m, l)
            elif g.subtype == 'G2':
                if fortran and FMODULES:
                    f_vects = np.array(vects)
                    value = fbp.fdg2(self.neighbor1, self.neighbor2, neighborindices, f_neighborsymbols, \
                    Rs, np.transpose(f_unitVects), index, m, l, angles_list, np.transpose(f_vects), \
                    np.transpose(xyzs), self.eta, self.r_s, self.r_cut, self.lambda_, self.zeta)
                else:
                    value = self.calculate_dG2(neighborindices, neighborsymbols, Rs, unitVects, index, m, l, angles_list, vects, xyzs)
            fingerprintprime[count] = value
        return fingerprintprime


    def calculate_dG1(self, neighborindices, neighborsymbols, Rs, unitVects, i, m, l):
        """
        BehlorParrinello.calculate_dG1()
        ===============================
        Function to calculate derivative of G1 for each atom

        Parameters
        ----------
        neighborindices : list of all neighbor indices for atom, calculated by NeighborList
        neighborsymbols : list of neighbor element types
        Rs : Rij values for each atom-neighbor pair
        unitVects : unit vectors for ij pairs
        i : index of center atom
        m : index of neighbor
        l : direction (x,y,z) = (0,1,2)
        """

        values = 0.

        for count in range(len(neighborindices)):
                neighborsymbol = neighborsymbols[count]
                j = neighborindices[count]
                Rij = Rs[count]
                Rij_uv = unitVects[count]
                if neighborsymbol == self.neighbor:
                    dRijdRml = dRij_dRml(i, j, Rij_uv, m, l)
                    if dRijdRml != 0:
                        values += ((dRijdRml * (-2*self.eta * (Rij - self.r_s) *  (0.5* (np.cos(np.pi * Rij / self.r_cut) + 1))/(self.r_cut**2)) + (0.5 * (-np.pi / self.r_cut) * np.sin(np.pi * Rij / self.r_cut) * dRijdRml )) * np.exp(-self.eta * ((Rij - self.r_s) ** 2) /(self.r_cut**2) ))

        return values


    def calculate_dG2(self, neighborindices, neighborsymbols, Rs, unitVects, i, m, l, angles_list, vects, xyzs):

        """
        BehlorParrinello.calculate_dG2()
        =========================
        Calculates derivatives of dG2 for each atom

        Parameters
        ----------
        neighborindices : list of all neighbor indices for atom, calculated by NeighborListi
        neighborsymbols : list of neighbor element types
        Rs : Rij values for each atom-neighbor pair
        unitVects : unit vectors for ij pairs
        i : index of center atom
        m : index of neighbor
        l : direction (x,y,z) = (0,1,2)
        angles_list : list of angles for all jik pairs
        vects : ij pair vectors 
        xyzs : coordinates of neighboratoms to calculate jk distances
        """

        # loop through each set of neighbors and calculate G2s for each triplet set

        value1 = 0.
        counts = range(len(neighborindices))
        q = 0
        for y in counts:
            for z in counts[(y+1):]:
                neigh1 = neighborsymbols[y]
                neigh2 = neighborsymbols[z]
                if neigh1 != self.neighbor1 or neigh2 != self.neighbor2:
                    q += 1
                    continue
                index_atom = i
                index_N1 = neighborindices[y]
                index_N2 = neighborindices[z]
                j = index_N1
                k = index_N2
                r_neighbor1 = Rs[y]
                r_neighbor2 = Rs[z]
                index_angle = angles_list[q]
                j_coord = xyzs[j]
                k_coord = xyzs[k]
                jk_vector = k_coord - j_coord
                jk_dist = np.linalg.norm(jk_vector)
                jk_unitVect = jk_vector/jk_dist

                vectorN1 = vects[y]
                vectorN2 = vects[z]
                unitV1 = unitVects[y]
                unitV2 = unitVects[z]

                dRij = dRij_dRml(i, j, unitV1, m, l)
                dRik = dRij_dRml(i, k, unitV2, m, l)
                dRjk = dRij_dRml(j, k, jk_unitVect, m, l)

                dCos = dCos_theta_ijk_dR_ml(index_atom, index_N1, index_N2, vectorN1, vectorN2, r_neighbor1, r_neighbor2, unitV1, unitV2, m, l)
                c1 = (1 + self.lambda_*np.cos(np.radians(index_angle)))

                term1 = c1 ** (self.zeta - 1.) * np.exp(-self.eta * (r_neighbor1**2 + r_neighbor2**2 + jk_dist**2)/self.r_cut**2)
                term2 = 0

                if dCos != 0:
                    term2 += self.lambda_ * self.zeta * dCos

                if dRij != 0:
                    term2 += -2. * c1 * self.eta * r_neighbor1 * dRij / self.r_cut**2
                if dRik != 0:
                    term2 += -2. * c1 * self.eta * r_neighbor2 * dRik / self.r_cut**2
                if dRjk != 0:
                    term2 += -2. * c1 * self.eta * jk_dist * dRjk / self.r_cut**2

                fcij = (0.5 * (np.cos(np.pi * r_neighbor1 / self.r_cut) + 1))
                fcik = (0.5 * (np.cos(np.pi * r_neighbor2 / self.r_cut) + 1))
                fcjk = (0.5 * (np.cos(np.pi * jk_dist / self.r_cut) + 1))

                term3 = (fcij * fcik * fcjk) * term2

                dfcij = (0.5 * (-np.pi / self.r_cut) * np.sin(np.pi * r_neighbor1 / self.r_cut) )
                dfcik = (0.5 * (-np.pi / self.r_cut) * np.sin(np.pi * r_neighbor2 / self.r_cut) )
                dfcjk = (0.5 * (-np.pi / self.r_cut) * np.sin(np.pi * jk_dist / self.r_cut) )

                term4 = dfcij * dRij * fcik * fcjk
                term5 = fcij * dfcik * dRik * fcjk
                term6 = fcij * fcik * dfcjk * dRjk
                value1 += term1 * (term3 + c1 * (term4 + term5 + term6))
                values = value1 * (2.**(1-self.zeta))
                q += 1

        return values


def dRij_dRml(i,j,unitV,m,l):
    if m == i:
        dRijdRml = -unitV[l]
    elif m == j:
        dRijdRml = unitV[l]
    else:
        dRijdRml = 0
    return dRijdRml


def Kronecker(i, j):
    """Kronecker delta function. Adapted from AMP

    Parameters
    ----------
    i : int
        First index of Kronecker delta.
    j : int
        Second index of Kronecker delta.

    Returns
    -------
    int
        The value of the Kronecker delta.
    """

    if i == j:
        return 1
    else:
        return 0

def dRij_dRml_vector(i, j, m, l):
    """Returns the derivative of the position vector R_{ij} with respect to
    x_{l} of itomic index m. Adapted from AMP.

    See Eq. 14d of the supplementary information of Khorshidi, Peterson, CPC(2016).

    Parameters
    ----------
    i : int
        Index of the first atom.
    j : int
        Index of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    list of float
        The derivative of the position vector R_{ij} with respect to x_{l} of atomic index m
    """

    if (m != i) and (m != j):
        return [0, 0, 0]
    else:
        dRij_dRml_vector = [None, None, None]
        c1 = Kronecker(m, j) - Kronecker(m, i)
        dRij_dRml_vector[0] = c1 * Kronecker(0, l)
        dRij_dRml_vector[1] = c1 * Kronecker(1, l)
        dRij_dRml_vector[2] = c1 * Kronecker(2, l)
        return dRij_dRml_vector


def dCos_theta_ijk_dR_ml(i, j, k, vectorN1, vectorN2, r_neighbor1, r_neighbor2, unitV1, unitV2, m, l):
    """Returns the derivative of Cos(theta_{ijk}) with respect to x_{l} of atomic index m. Adapted from AMP.

    See Eq. 14f of the supplementary information of Khorshidi, Peterson, CPC(2016).

    Parameters
    ----------
    i : int
        Index of the center atom.
    j : int
        Index of the first atom.
    k : int
        Index of the second atom.
    Ri : float
        Position of the center atom.
    Rj : float
        Position of the first atom.
    Rk : float
        Position of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    dCos_theta_ijk_dR_ml : float
        Derivative of Cos(theta_{ijk}) with respect to x_{l} of atomic index m.
    """

    Rij_vector = vectorN1
    Rij = r_neighbor1
    Rik_vector = vectorN2
    Rik = r_neighbor2
    dCos_theta_ijk_dR_ml = 0

    dRijdRml = dRij_dRml_vector(i,j,m,l)
    if np.array(dRijdRml).any() != 0:
        dCos_theta_ijk_dR_ml += np.dot(dRijdRml, Rik_vector) / (Rij * Rik)

    dRikdRml = dRij_dRml_vector(i,k,m,l)
    if np.array(dRikdRml).any() != 0:
        dCos_theta_ijk_dR_ml += np.dot(Rij_vector, dRikdRml) / (Rij * Rik)

    dRijdRml = dRij_dRml(i,j,unitV1,m,l)
    if dRijdRml != 0:
        dCos_theta_ijk_dR_ml += - np.dot(Rij_vector, Rik_vector) * dRijdRml / ((Rij ** 2.) * Rik)
    dRikdRml = dRij_dRml(i,k,unitV2,m,l)
    if dRikdRml != 0:
        dCos_theta_ijk_dR_ml += - np.dot(Rij_vector, Rik_vector) * dRikdRml / (Rij * (Rik ** 2.))

    return dCos_theta_ijk_dR_ml


def represent_BP(nl, nFPs, trainingimages, properties, num_batch=1,  G_paras=None,
                 fpfilename='fps.pckl', fortran=False, normalize=True, fNN=False, fpDir= None, logger=None): #ZB ADDED: fpDir= None
    starttime = time.time()
    """
    represent_BP
    ===========
    computes the Behler-Parrinello atom-based descriptors for each atom in a given structure

    Parameters
    ----------
    nl: neighborlist
    nFPs: number of fingerprints for each atom
    trainingimages: dictionary of image indices
    properties: energies and forces for each image
    num_batch: number of batches to split images into for fingerprint calculations
      Default =1. It is faster to use 1 batch, but if memory is an issue, you can temporailly store in multiple batches
    G_paras: parameters for BP fingerprints
    fortran: Use fortran FP calculator. default = False

    Returns
    -------
    fingerprints:
        (# atoms, # unique element types, # descriptors) array of G^1s and
        (# atoms, # unique element type pairs, # descriptors) array of G^2s
        If derivs=True, also returns
        (# atoms, # atoms, # cart directions(3), # unique element types,
        # descriptors) array of dG^1/dRs and
        (# atoms, # atoms, # cart directions(3), # unique element type
        pairs, # descriptors) array of dG^2/dRs
    interaction_dims: Dimensionality of interaction(s).
        (e.g. 1 for pairwise, 2 for triplets, [1,2] for both)

    fp: {'fpRange':fpRange, 'fpData': {image:<fp_object>, image2:<fp_object2> ...}}

    Notes
    -----
    Behler-Parrinello symmetry functions as described in:
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401
    """

    if num_batch > 1:
        tmp = tempfile.TemporaryDirectory()
        bdict = {}
        bsize = math.ceil(len(trainingimages)/num_batch)
        for b in range(bsize):
            bdict[b] = []
        ar = np.arange(len(trainingimages))
        rem = ((bsize*num_batch)%len(trainingimages))
        blist_temp = np.pad(ar, ((0,rem)),'constant', constant_values=np.nan)
        blist = np.reshape(blist_temp, (-1, num_batch))

    ### ZB ADD START
    if fpDir is None:
        fpsdir = os.getcwd() + '/fingerprints'
    else:
        fpsdir = fpDir
    if not os.path.exists(fpsdir):
        os.mkdir(fpsdir)
    ### ZB ADD END
    bcount = 0

    fpDb = {}
    fpDerDb = {}
    fortran_fpDerDb = {}
    fpData = OrderedDict() 
    fpData_temp = {}
    aEfps = {}
    for ele in nFPs.keys():
        aEfps[ele] = []
    fptime = 0
    for struct in trainingimages.keys():
        if logger and struct % 20 == 0:
            logger.info('  Calculating FPs for Image %d', struct)
        # get FPs and FPprimes for each structure
        st = time.time()
        if fNN:
            fingerprints, fingerprintprimes, fortran_fpprimes = loop_FPs(struct, nl, nFPs, G_paras, fortran, fNN)
            #print ('fingerprints')
            #print (fingerprints)
        else:
            fingerprints, fingerprintprimes = loop_FPs(struct, nl, nFPs, G_paras, fortran, fNN)
        fptime += time.time()-st
        #print('fingerprints', fingerprints)
        #for fpkey in fingerprintprimes.keys():
        #  print(fpkey, fingerprintprimes[fpkey])
        # Temporary change for active learning
        #if len(trainingimages) == 1:
        #   struct = fpKey

        # Save FPs and FPprimes to dictionary 
        fpDb[(struct)] = fingerprints
        fpDerDb[struct] = fingerprintprimes
        acf = atomCenteredFPs()
        chemsymbols = trainingimages[struct].get_chemical_symbols() #ZB ADDED!
        if fNN:
            fortran_fpDerDb[struct] = fortran_fpprimes
        # For pyamff calculator
        if properties is None:
            p1 = None
            acf.sortFPs(fpDb, nFPs, p1, [struct], fpDerDb, batch=False)
            if fNN:
                acf.sort_fortranFPprimes([struct], fortran_fpDerDb)
            return acf

        # Store FPs and FPprimes as acf objects and make readable by pytorch/fortran machine learning
        p1 = {struct: properties[struct]}
        formatedfingerprints=[]
        for i in range(len(fingerprints)):
             temp = list(fingerprints[i])
             #temp = fingerprints[i]
             del temp[0]
             formatedfingerprints.append(temp[0])
             #print("temp",temp)
        #print("formated fps",formatedfingerprints)
        #print("fingerprints behler 667-->", fingerprints)
        #print("fpDb behler 668-->", fpDb)
        #fingerprints = [[0.92011],[0.92011]]
        #acf.sortFPs(fpDb, nFPs, p1, [struct], fpDerDb) ###ORIGINAL #ZB 
        #acf.sortFPs(chemsymbols,fingerprints, nFPs, p1, [struct], fingerprintprimes) #ZB
        acf.sortFPs(chemsymbols,formatedfingerprints, nFPs, p1, [struct], fingerprintprimes) #ZB ADDED FORMATEDFPS!

        # If saving FPs in batches:
        if num_batch > 1:
            xx = np.where(blist==bcount)
            bind = xx[0][0]
            breset = xx[1][0]
            if breset == 0:
                fpData_temp = {}

            fpData_temp[struct] = acf
            bdict.update(fpData_temp)
            f_name = os.path.join(tmp.name, 'batch_{}.pckl'.format(bind))

            with bz2.BZ2File(f_name, 'wb') as f:
                pickle.dump(bdict,f)

            if breset == 0:
                bdict = {}
            bcount +=1
        # If storing all FPs in one pickle file:
        else:
            fpData[struct] = acf 

        for k,v in acf.allElement_fps.items():
            aEfps[k].extend(v)

    # Recombine all batches into one dictionary
    if num_batch > 1:
        for filename in os.listdir(tmp.name):
            fname = os.path.join(tmp.name, filename)
            with bz2.BZ2File(fname, 'rb') as f1:
                data = pickle.load(f1)
                fpData.update(data)
    fpData[struct] = acf
                #print(acf.allElement_fps)
    ## ZB ADD START
    # Copied from fingerprints.py
    if struct%num_batch == 0 or struct == len(trainingimages)-1:
        for key in fpData.keys():
            #print ("=725 behpar.py fpData.keys(): ",fpData.keys())
            f_name = fpsdir+'/fps_{}.pckl'.format(key)
            with bz2.BZ2File(f_name, 'wb') as f:
               pickle.dump(fpData[key],f)
        fpData = {}
    # ZB ADD END
    # Get range for FPs
    fprange = {}
    if normalize:
        for ele in nFPs.keys():
            minv = torch.min(torch.stack(aEfps[ele]), dim=0)[0]
            maxv = torch.max(torch.stack(aEfps[ele]), dim=0)[0]
            fprange[ele] = [minv, maxv, maxv-minv]

        fprange, magnitudeScale, interceptScale = normalizeParas(fprange)
        for key in fpData.keys():
            fpData[key].normalizeFPs(fprange, magnitudeScale, interceptScale)

    fp = {'fprange':fprange, 'fpData': fpData}
    #print('TIMEUSED', time.time()-starttime)
    print('Fingerprint time: %8.2f s', fptime)

    # Save fp dictionary with fprange and fpdata to pickle file. 'fps.pckl' is default file name
    #io.save_data(fp, fpfilename) Omitted! #ZB

    # Do we still need to return these?? RAC
    return fpData, fprange, magnitudeScale, interceptScale

def loop_FPs(struct, nl, nFPs, G_paras, fortran, fNN):

    bp = BehlerParrinello()

    fingerprints = []
    fingerprintprimes = {}
    #fortran fp primes structure: max(nGs), 3, max(nneighbors)+1, total_natoms
    fortran_fpprimes = np.zeros((max(list(nFPs.values())),3,max(map(len,nl.nlist[struct]))+1,len(nl.nlist[struct])))
    # Temporary change for active learning
    #if len(trainingimages) == 1:
    #    fpKey = struct
    #    struct = 0
    for atom in range(0, len(nl.nlist[struct])):
        xyzs = nl.coords[struct]
        selfsymbol = nl.symbols[struct][atom]
        selfindex = atom
        selfneighborindices = nl.nlist[struct][atom]
        selfneighborsymbols = nl.neighborSymbols[struct][atom]
        selfneighboroffsets = nl.offsets[struct][atom]
        Rs_self = nl.dists[struct][atom]
        unitVects_self = nl.unitVects[struct][atom]
        angles_self = nl.angles[struct][atom]
        vects_self = nl.vects[struct][atom]

        fp = bp.get_fp(selfsymbol, selfneighborindices, selfneighborsymbols,
                       Rs_self, G_paras, angles_self, xyzs, fortran)
        fingerprints.append(fp)

        for l in range(3):
            fpprime = bp.get_fp_primes(selfindex, selfsymbol,
                                       selfneighborindices, selfneighborsymbols,
                                       Rs_self, unitVects_self,
                                       selfindex, l, G_paras, angles_self, vects_self, xyzs, fortran)
            fingerprintprimes[(selfindex, selfsymbol, selfindex, selfsymbol, l)] = fpprime
            #print ('selfindex, l', selfindex, l)
            #print (fpprime)
            fortran_fpprimes[0:nFPs[selfsymbol],l,0,selfindex] = fpprime
            nnindex = 1
            for nindex, nsymbol, noffset in zip(selfneighborindices, selfneighborsymbols, selfneighboroffsets):
                if noffset.all() == 0:
                    nneighborindices = nl.nlist[struct][nindex]
                    nneighborsymbols = nl.neighborSymbols[struct][nindex]
                    Rs_neighbor = nl.dists[struct][nindex]
                    unitVects_neighbor = nl.unitVects[struct][nindex]
                    angles_neighbor = nl.angles[struct][nindex]
                    vects_neighbor = nl.vects[struct][nindex]
                    fpprime = bp.get_fp_primes(nindex, nsymbol, nneighborindices, nneighborsymbols,
                                               Rs_neighbor, unitVects_neighbor, selfindex, l,
                                               G_paras, angles_neighbor, vects_neighbor, xyzs, fortran)

                    fingerprintprimes[(selfindex, selfsymbol, nindex, nsymbol, l)] = fpprime
                    if fNN:
                        #print ('neighbor', nnindex, 'of atom', selfindex)
                        #print (fpprime)
                        fortran_fpprimes[0:nFPs[selfsymbol],l,nnindex,selfindex] = fpprime
                nnindex += 1
    if fNN:
        return fingerprints, fingerprintprimes, fortran_fpprimes
    else:
        return fingerprints, fingerprintprimes


def loop_FPs_precon(struct, nl, G_paras, fortran):

    bp = BehlerParrinello()

    fingerprints = {}
    #fp_dict = {}
    fingerprintprimes = {}
    # Tempory change for active learning
    #if len(trainingimages) == 1:
    #    fpKey = struct
    #    struct = 0
    for atom in range(0, len(nl.nlist[struct])):
        xyzs = nl.coords[struct]
        selfsymbol = nl.symbols[struct][atom]
        selfindex = atom
        selfneighborindices = nl.nlist[struct][atom]
        selfneighborsymbols = nl.neighborSymbols[struct][atom]
        selfneighboroffsets = nl.offsets[struct][atom]
        Rs_self = nl.dists[struct][atom]
        unitVects_self = nl.unitVects[struct][atom]
        angles_self = nl.angles[struct][atom]
        vects_self = nl.vects[struct][atom]

        #fp = bp.get_fp_precon(selfsymbol,\
        #      selfneighborindices, selfneighborsymbols, \
        #      Rs_self, \
        #      G_paras, angles_self, xyzs, atom, fortran)
        #fingerprints[atom] = fp

        for l in range(3):
            fp = bp.get_fp_precon(selfsymbol, selfneighborindices, selfneighborsymbols,
                                  Rs_self, G_paras, angles_self, xyzs, atom, fortran, l)
            fingerprints[(3*atom)+l] = fp

            fpprime = bp.get_fp_primes_precon(selfindex, selfsymbol, selfneighborindices, selfneighborsymbols,
                                              Rs_self, unitVects_self, selfindex, l, G_paras, angles_self, vects_self, xyzs, fortran)
            c = (selfindex*3) +l
            n = (((selfindex*3)+l)) # GH: c and n are the same? And why all the brackets?
            fingerprintprimes[(c, n)] = fpprime
            #fingerprints[(c,n)] = fp
            #fingerprints[atom] = fp
            for nindex, nsymbol, noffset in zip(selfneighborindices, selfneighborsymbols, selfneighboroffsets):
                if noffset.all() == 0:
                    nneighborindices = nl.nlist[struct][nindex]
                    nneighborsymbols = nl.neighborSymbols[struct][nindex]
                    Rs_neighbor = nl.dists[struct][nindex]
                    unitVects_neighbor = nl.unitVects[struct][nindex]
                    angles_neighbor = nl.angles[struct][nindex]
                    vects_neighbor = nl.vects[struct][nindex]
                    fpprime = bp.get_fp_primes_precon(nindex, nsymbol, nneighborindices, nneighborsymbols, Rs_neighbor, unitVects_neighbor,
                                                      selfindex, l, G_paras, angles_neighbor, vects_neighbor, xyzs, fortran)
                    #fp = bp.get_fp_precon(selfsymbol,\
                              #selfneighborindices, selfneighborsymbols, \
                              #Rs_self, \
                              #G_paras, angles_self,  xyzs, atom, fortran)

                    #c = (selfindex)*3
                    n = nindex*3 + l
                    fingerprintprimes[(c, n)] = fpprime
                    #fingerprints[(c,n)] = fp
    return fingerprints, fingerprintprimes

