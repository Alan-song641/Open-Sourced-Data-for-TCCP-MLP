import numpy as np
from ase.neighborlist import NeighborList
from ase import Atoms
import ase
#try:
#    from pyamff.fmodules import fnlist
#    print ("Fortran NLIST loaded")
#    FMODULES = True
#except:
#    FMODULES = False
#    print ("Fortran NLIST not loaded")

"""
.. module:: NeighborLists
    :members: NeighborLists

returns: neighborlist object

"""


class NeighborLists:
    """
    Class which contains neighbor list information, including Rs, angles and vectors

    """

    def __init__(self, cutoff):
        #self.fingerprints = fingerprints
        self.cutoff = cutoff / 2.0

    def calculate(self, images, fortran=False):
        """
        Calculates Neighborlist information

        """

        # Big dictionary containing all data
        data = {}

        elements = None
        index = 1
        self.unitVects = []
        self.vects = []
        self.dists = []
        self.nlist = []
        self.offsets = []
        self.symbols = []
        self.neighborSymbols = []
        self.angles = []
        self.coords = []

        """
        {0: nl of image 0,
         1: nl of image 1}
        """
        for i in range(len(images)):
            self.unitVects.append([])
            self.dists.append([])
            self.vects.append([])
            self.nlist.append([])
            self.offsets.append([])
            self.symbols.append([])
            self.neighborSymbols.append([])
            self.angles.append([])

        a_index = 0
        for key in images.keys():
            atoms = images[key]
            if(fortran and FMODULES):
                # copy data to python data structures
                self.coords.append(atoms.get_positions())
                self.loop_images_fortran(atoms, a_index)

            else:

                """# Create Neighborlist for each structure
                nl = NeighborList(cutoffs=[r_cut_2]*len(atoms), sorted=False, self_interaction=False, bothways=True, skin=0.)
                nl.update(atoms)
                self.coords.append(atoms.get_positions())
                for i in range(0,len(atoms)):
                    indices,offsets = nl.get_neighbors(i)
                    self.nlist[a_index].append(np.sort(indices))
                    #self.nlist[a_index].append(indices)
                    self.offsets[a_index].append(offsets)

                # genetate array for pair data"""

                self.loop_images(atoms, a_index)

            a_index += 1


    def loop_images_fortran(self, atoms, a_index):

        max_neighs = 48  # currently hard coded to avoid passing dynamic arrays
        max_angles = 1024
        num_atoms = len(atoms)
        num_neigh = np.zeros(num_atoms, dtype=np.dtype('i4'))
        num_angle = np.zeros(num_atoms, dtype=np.dtype('i4'))
        neighs = np.zeros([num_atoms,max_neighs], dtype=np.dtype('i4'), order='F')
        dists = np.zeros([num_atoms, max_neighs], order='F')
        vects = np.zeros([num_atoms, max_neighs, 3], order='F')
        unitvects = np.zeros([num_atoms, max_neighs, 3], order='F')
        angles = np.zeros([num_atoms, max_angles], order='F')

        fnlist.rcut = self.cutoff*2.0
        fnlist.calc(atoms.get_positions(), atoms.get_cell(),
                    num_neigh, num_angle, neighs, dists, vects, unitvects, angles)

        for i in range(len(atoms)):
            nlist = neighs[i,0:num_neigh[i]] - 1  # fortran -> python index change
            self.nlist[a_index].append(nlist.astype(int))
            self.offsets[a_index].append(np.zeros([num_neigh[i],3],dtype=int))
            symbols = []
            vects_tmp = []
            unitvects_tmp = []
            neighborSymbols = []
            for j in range(0,len(self.nlist[a_index][i])):
                neighborSymbols.append(atoms.symbols[self.nlist[a_index][i][j]])
                vects_tmp.append(vects[i][j])
                unitvects_tmp.append(unitvects[i][j])
            self.symbols[a_index].append(atoms.symbols[i])
            self.dists[a_index].append(dists[i,0:num_neigh[i]].tolist())
            #self.vects[a_index].append(vects[i,0:num_neigh[i]])
            #self.unitVects[a_index].append(unitvects[i,0:num_neigh[i]])
            self.vects[a_index].append(vects_tmp)
            self.unitVects[a_index].append(unitvects_tmp)
            self.neighborSymbols[a_index].append(neighborSymbols)
            self.angles[a_index].append(angles[i,0:num_angle[i]])


    def loop_images(self, atoms, a_index):
        r_cut_2 = self.cutoff # radial cutoff / 2

        nl = NeighborList(cutoffs=[r_cut_2]*len(atoms), sorted=False, self_interaction=False, bothways=True, skin=0.)
        nl.update(atoms)
        self.coords.append(atoms.get_positions())
        for i in range(0,len(atoms)):
            indices,offsets = nl.get_neighbors(i)
            self.nlist[a_index].append(np.sort(indices))
            #self.nlist[a_index].append(indices)
            self.offsets[a_index].append(offsets)

        for i in range(0,len(self.nlist[a_index])):
            dists = []
            vects = []
            unit_vects = []
            symbols = []
            neighborSymbols = []
            coords = []
            for j in range(0,len(self.nlist[a_index][i])):
                distance = atoms.get_distance(i, self.nlist[a_index][i][j], mic=True)
                dists.append((distance))
                vector = atoms.get_distance(i, self.nlist[a_index][i][j], mic=True, vector=True)
                vects.append(vector)
                unit_vector = vector / distance
                unit_vects.append(unit_vector)
                neighborSymbols.append(atoms.symbols[self.nlist[a_index][i][j]])
            self.symbols[a_index].append(atoms.symbols[i])
            self.dists[a_index].append(dists)
            self.vects[a_index].append(vects)
            self.unitVects[a_index].append(unit_vects)
            self.neighborSymbols[a_index].append(neighborSymbols)

        # Generate angle data
        for aa in range(0,len(self.nlist[a_index])):
            angles = []
            for aaa in range(0,len(self.nlist[a_index][aa])):
                for aaaa in range(aaa+1,len(self.nlist[a_index][aa])):
                    angs = atoms.get_angle(self.nlist[a_index][aa][aaa], aa, self.nlist[a_index][aa][aaaa], mic=True)
                    angles.append(angs)

            self.angles[a_index].append(angles)

