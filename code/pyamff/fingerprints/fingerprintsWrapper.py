import numpy as np
import torch
from torch import cat
import copy
#from memory_profiler import profile

import warnings
warnings.filterwarnings("ignore")

class atomCenteredFPs(object):

    def __init__(self):
        self.elements = None
        self.allElement_fps = {}           # Fingerprints for all images {'Au':array([[]]]}
        self.fp_imageIndices = {}
        self.original_fps = None           # Fingerprints for all images {'Au':array([[]]]}
        self.original_dgdx = None
        self.original_fortran_dgdx = None
        self.fpRange = {}
        self.dgdx = {}
        self.dEdg_AtomIndices = {}
        self.force_AtomIndices = {}
        self.natomsPerImageFxyz = None     # used for force calculation when using cohesive energy as target
        self.natomsPerElement = None       # {'H':2, 'Pd': 13}
        self.natomsPerImageForce = None    # used for loss calculation
        self.natomsPerImageEnergy = None   # used for loss calculation
        self.ntotalAtoms = None            # a value: total number of atoms
        self.nimages = None                # a value: total number of images
        self.energies = []
        self.forces = []
        self.indices = []
        self.original_force_AtomIndices = None
        self.fortran_dgdx = None           # Fortran fp_primes
        self.accN = {}

    # For on-the-fly training
    def setOriginal(self):
        self.original_fps = copy.deepcopy(self.allElement_fps)
        self.original_dgdx = copy.deepcopy(self.dgdx)
        self.original_fortran_dgdx = copy.deepcopy(self.fortran_dgdx)
        self.original_energies = copy.copy(self.energies)
        self.original_forces = copy.copy(self.forces)

    def fetchOriginal(self):
        self.allElement_fps = copy.deepcopy(self.original_fps)
        self.dgdx = copy.deepcopy(self.original_dgdx)
        self.fortran_dgdx = copy.deepcopy(self.original_fortran_dgdx)
        self.energies = copy.copy(self.original_energies)
        self.forces = copy.copy(self.original_forces)

    def scaleEnergies(self):
        if len(self.energies) <= 1:
            intercept = 0.
            slope = 1.
            #sys.stderr.write('Number of images is required to be larger than 1 but %d is given\n'%(len(self.energies)))
            #sys.exit(2)
        else:
            #intercept = np.mean(self.energies)
            intercept = self.energies.mean()
            self.energies = self.energies - intercept
            slope = torch.abs(self.energies).mean()
        return slope, intercept

    def findFPrange(self):
        #for element in self.allElement_fps.keys():
        #    minv = np.min(self.allElement_fps[element], axis=0)
        #    maxv = np.max(self.allElement_fps[element], axis=0)
        #    self.fprange[element] = [minv, maxv, maxv-minv]
        for element in self.allElement_fps.keys():
            minv = torch.min(self.allElement_fps[element], dim=0)[0]
            maxv = torch.max(self.allElement_fps[element], dim=0)[0]
            self.fprange[element] = [minv, maxv, maxv-minv]

    def normalizeFPs(self, fprange, magnitudeScale, interceptScale):
        elements = list(fprange.keys())
        for element in elements:
            total_atoms = len(self.allElement_fps[element])
            # print('mag', magnitudeScale[element])
            mags = magnitudeScale[element].repeat(total_atoms,1)
            # print('fprange')
            # print(fprange)
            inters = interceptScale[element].repeat(total_atoms,1)
            #print ('inters')
            #print (inters)
            if len(self.allElement_fps[element]) != 0:
                self.allElement_fps[element] = (torch.mul(self.allElement_fps[element], mags) + inters).double()
                self.allElement_fps[element].requires_grad = True
                
                if len(self.dgdx[element]) != 0:
                    self.dgdx[element] = torch.mul(self.dgdx[element], torch.flatten(mags)[self.dEdg_AtomIndices[element]])
                    self.dgdx[element].requires_grad = True

    #from memory_profiler import profile
    #@profile
    #batchgenerator calls stackFPs as a collate function, I dont know we need it?
    #everything in here is a tensor already except self.elements and elements
    def stackFPs(self, acfs, new=True, activelearning=False):
        i = 0
        #if self.original_force_AtomIndices is not None:
        #   self.force_AtomIndices = copy.copy(self.original_force_AtomIndices)
        for acf in acfs:
            if i == 0:
                self.elements = list(acf.allElement_fps.keys())
                elements = list(acf.allElement_fps.keys())
            if new:
                self.allElement_fps = copy.copy(acf.allElement_fps)
                self.fp_imageIndices = copy.copy(acf.fp_imageIndices)
                self.dgdx = copy.copy(acf.dgdx)
                self.dEdg_AtomIndices = copy.copy(acf.dEdg_AtomIndices)
                self.force_AtomIndices = copy.copy(acf.force_AtomIndices)
                self.natomsPerElement = copy.copy(acf.natomsPerElement)
                self.natomsPerImageForce = acf.natomsPerImageForce
                self.natomsPerImageEnergy = acf.natomsPerImageEnergy
                self.ntotalAtoms = acf.ntotalAtoms
                self.nimages = acf.nimages
                self.energies = acf.energies
                self.forces = acf.forces
                self.fortran_dgdx = copy.copy(acf.fortran_dgdx)
                new = False
            else:
                i+=1
                for element in elements:
                    # Cat two tensor maybe memory consuming computation order matters
                    #print('force_AtomIndices', self.force_AtomIndices[element])
                    #print('dtype', acf.force_AtomIndices[element].dtype)
                    #print(acf.dgdx[element].dtype)
                    self.force_AtomIndices[element] = cat([self.force_AtomIndices[element],
                                                           acf.force_AtomIndices[element] + self.ntotalAtoms])
                    self.natomsPerElement[element] += acf.natomsPerElement[element]
                    #print('dEdg_AtomInd', self.dEdg_AtomIndices[element])
                    self.dEdg_AtomIndices[element] = cat([self.dEdg_AtomIndices[element],
                                                          acf.dEdg_AtomIndices[element] + self.allElement_fps[element].numel()
                                                         ])
                    #print('fpImageIndices', self.fp_imageIndices[element])
                    self.fp_imageIndices[element] = cat([self.fp_imageIndices[element], acf.fp_imageIndices[element]+i])
                    self.allElement_fps[element] = cat([self.allElement_fps[element], acf.allElement_fps[element]])
                    self.dgdx[element]  = cat([self.dgdx[element], acf.dgdx[element]])
                self.ntotalAtoms += acf.ntotalAtoms
                self.nimages += acf.nimages
                self.energies = cat([self.energies, acf.energies])
                self.forces = cat([self.forces, acf.forces])
                self.natomsPerImageForce = cat([self.natomsPerImageForce, acf.natomsPerImageForce])
                self.natomsPerImageEnergy = cat([self.natomsPerImageEnergy,acf.natomsPerImageEnergy])
        forceIndices = None
        if activelearning:
            self.original_force_AtomIndices = copy.copy(self.force_AtomIndices)
        for element in elements:
            """
            if forceIndices is None:
                forceIndices = self.force_AtomIndices[element]
            else:
                forceIndices = torch.cat([forceIndices, self.force_AtomIndices[element]])
            """

            if forceIndices is None:
                try:
                    forceIndices = self.force_AtomIndices[element]
                except:
                    forceIndices = self.force_AtomIndices
            else:
                try:
                    forceIndices = torch.cat([forceIndices, self.force_AtomIndices[element]])
                except:
                    #forceIndices = torch.cat([forceIndices, self.force_AtomIndices[element]])
                    forceIndices = self.force_AtomIndices

        self.force_AtomIndices = forceIndices
        self.natomsPerImageFxyz = torch.reshape(self.natomsPerImageForce,(self.ntotalAtoms, 1)).repeat(1,3)
        #print(self.natomsPerElement)
    
    #toTensor is called by preprocess in datapartitioner to add device before dumping
    #batchgenerator calls stackFPs again as a collate function, I dont know we need to call stackFPs twice.
    def toTensor(self, device=None): 
        d = device
        for element in self.elements:
            self.dEdg_AtomIndices[element] = self.dEdg_AtomIndices[element].to(device=d)
            self.fp_imageIndices[element] = self.fp_imageIndices[element].to(device=d)
            self.allElement_fps[element] = self.allElement_fps[element].to(device=d)
            self.dgdx[element] = self.dgdx[element].to(device=d)

        self.force_AtomIndices = self.force_AtomIndices.to(device=d)
        self.energies = self.energies.to(device=d)
        self.forces = self.forces.to(device=d)
        self.natomsPerImageForce = self.natomsPerImageForce.to(device=d)
        self.natomsPerImageEnergy = self.natomsPerImageEnergy.to(device=d)


    def sortFPs(self, atomSymbols, fingerprintDB, elementFPs,
                properties=None, keylist=None,
                fingerprintDerDB=None, fortran_fpprimesDB=None, batch=True):
        """
        This function generates the inputs to the tensorflow graph for the selected images.
        The essential problem is that each neural network is associated with a specific element type.
        Thus, atoms in each ASE image need to be sent to different networks.

        Inputs:
        fingerprintDB: a database of fingerprints, as taken from the descriptor
        elementFPs: an ordered dictionary of number of fingerprints for each type of element (e.g. {'C':2,'O':5}, etc)
        keylist: a list of hashs into the fingerprintDB for which we want to create inputs
        fingerprintDerDB: a database of fingerprint derivatives, as taken from the descriptor

        Outputs:
        allElement_fps: a dictionary of fingerprint inputs to each element's neural
            network
            Note: G_H_1 represent the 1st fingerprint centered on 'H'
                  Assume we have g1 fingerprints that are centered on 'H',
                                 g2 fingerprints that are centered on 'Pd'
            {'H':tensor([ [G_H_1,G_H_2,...,G_H_g1],  Atom 1  in Image 1
                          [G_H_1,G_H_2,...,G_H_g1],  Atom 2  in Image 1
                                 ...
                          [G_H_1,G_H_2,...,G_H_g1],  Atom N1 in Image 1

                          [G_H_1,G_H_2,...,G_H_g1],  Atom 1  in Image 2
                                 ...
                          [G_H_1,G_H_2,...,G_H_g1],  Atom N2 in Image 2
                                 ...
                          [G_H_1,G_H_2,...,G_H_g1],  Atom NM in Image M
                         ])
             'Pd':tensor([[G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom 1  in Image 1
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom 2  in Image 1
                                 ...
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom N1 in Image 1

                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom 1  in Image 2
                                 ...
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom N2 in Image 2
                                 ...
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom NM in Image M
                         ])
              }
        ntotalAtoms: the total number of atoms in the training batch

        dgdx: dictionary of fingerprint derivatives. Grouped by elements:          contrib. From      contrib. TO
             {'H': ([  [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Ggx, Ggy, Ggz]],        neighbor 1           atom 1
                         -----------     ------------       --------------
                            1st fp         2nd fp              gth fp
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor 2           atom 1
                                               ...                                     ...
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor M1          atom 1
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor 1           atom 2
                                               ...                                     ...
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor M2          atom 2
                                                .                                       .                  .
                                                .                                       .                  .
                                                .                                       .                  .

                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor 1           atom N
                                               ...                                     ...
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor MN          atom N
                           (Note: N is total number of 'element' atoms in training images)

                               ] ),
                            'Pd': ()
                           },
             }
        """
        fingerprintDB = torch.from_numpy(fingerprintDB)

        elements = elementFPs.keys()
        self.nimages = len(keylist)
        self.allElement_fps = {}
        allAtomIndices = {}  # store the location of atom in the whole tensor
        allfpIndices = {}  # store index of each fingerpint in allElement_fps
        natoms = []
        natomsPerImage = []
        self.fprange = {}
        self.fp_imageIndices = {}
        for element in elements:
            self.allElement_fps[element] = []
            self.fp_imageIndices[element] = []
        # print(allElement_fps.keys())
        tlocation = 0
        for j in range(len(keylist)):
            # For pyamff calculator
            if properties is not None:
                self.energies.extend(properties[keylist[j]][0])
                self.forces.extend(properties[keylist[j]][1])

            nElement = {}
            allfpIndices[keylist[j]] = []
            allAtomIndices[keylist[j]] = []
            natom = len(atomSymbols)

            for i in range(len(atomSymbols)):
                self.allElement_fps[atomSymbols[i]].append(fingerprintDB[i][:elementFPs[atomSymbols[i]]])
                currlocation = len(self.allElement_fps[atomSymbols[i]])
                allAtomIndices[keylist[j]].append(tlocation)
                allfpIndices[keylist[j]].append([index for index in \
                                                 range((currlocation - 1) * elementFPs[atomSymbols[i]], 
                                                       currlocation * elementFPs[atomSymbols[i]])])
                self.fp_imageIndices[atomSymbols[i]].append([j])
                tlocation += 1
                if atomSymbols[i] not in nElement:
                    nElement[atomSymbols[i]] = 1
                else:
                    nElement[atomSymbols[i]] += 1
                natomsPerImage.append(natom)
            natoms.append(natom)

        self.energies = torch.tensor(self.energies, dtype=torch.double)
        self.forces = torch.from_numpy(np.array(self.forces, dtype=np.float64))

        self.natomsPerElement = {}
        ntotalAtoms = tlocation
        for element in elements:
            self.natomsPerElement[element] = len(self.allElement_fps[element])
            if len(self.allElement_fps[element]) == 0:
                self.allElement_fps[element] = torch.empty((0,))
                # self.fprange[element] = [0, 0, 0]
                continue
            #   minv = np.array([100.]*elementFPs[element])
            #   maxv = np.array([0.]*elementFPs[element])
            else:
                self.allElement_fps[element] = torch.stack(self.allElement_fps[element])
                minv = torch.amin(self.allElement_fps[element], axis=0)
                maxv = torch.amax(self.allElement_fps[element], axis=0)
                self.fprange[element] = [minv, maxv, maxv - minv]

        # Set up the array for atom-based fingerprint derivatives.
        self.dgdx = {}
        self.force_AtomIndices = {}  # Used to sum forces over atoms
        self.dEdg_AtomIndices = {}  # Used to fetch dEdg to be used to multiply with dgdx tensor
        # dict_init = {}
        
        for element in elements:
            self.dgdx[element] = []
            self.force_AtomIndices[element] = []
            # dict_init[element] = []
            self.dEdg_AtomIndices[element] = []

        for j in range(len(keylist)):
            # Iterate over all atoms in the image
            natom = natoms[j]
            # dgdx_image = copy.deepcopy(dict_init)

            '''----------optimized part---------------'''
            for wrtIndex in range(natom):  # make sure images are ordered by element and elementFPs
                wrtSymbol = atomSymbols[wrtIndex]
                # TODO: iterate over neighborlist only
                for centerIndex in range(natom):
                    centerSymbol = atomSymbols[centerIndex]
                    key = (wrtIndex, wrtSymbol, centerIndex, centerSymbol, 0) # if key with direction = 0 exists, it also exist for direction = 1 and 2
                    if key in fingerprintDerDB.keys():
                        dgdx_temp = np.empty((3, elementFPs[centerSymbol]))
                        dgdx_temp[0, :] = fingerprintDerDB[(wrtIndex, wrtSymbol, centerIndex, centerSymbol, 0)]
                        dgdx_temp[1, :] = fingerprintDerDB[(wrtIndex, wrtSymbol, centerIndex, centerSymbol, 1)]
                        dgdx_temp[2, :] = fingerprintDerDB[(wrtIndex, wrtSymbol, centerIndex, centerSymbol, 2)]

                        if len(dgdx_temp) > 1: # there may exist an atom with 0 neighbor
                            self.dgdx[centerSymbol].append(torch.from_numpy(dgdx_temp).T)
                            self.dEdg_AtomIndices[centerSymbol].append(allfpIndices[keylist[j]][centerIndex])
                            self.force_AtomIndices[centerSymbol].append([allAtomIndices[keylist[j]][wrtIndex]] * 3)
            '''-------------------------'''

        forceIndices = None
        for element in elements:
            # print("allelementSort",self.allElement_fps[element])
            # GH self.allElement_fps[element] = torch.tensor(self.allElement_fps[element])
            # self.allElement_fps[element] = torch.stack(self.allElement_fps[element])
            # self.dgdx[element] = torch.stack(self.dgdx[element])
            if len(self.dgdx[element]) != 0:
                self.dgdx[element] = torch.stack(self.dgdx[element])
            else:
                self.dgdx[element] = torch.empty((0,))
            # print('dedg_AtomINd',self.dEdg_AtomIndices[element])

            self.dEdg_AtomIndices[element] = torch.tensor(self.dEdg_AtomIndices[element], dtype=torch.int64).\
                                            view(len(self.dEdg_AtomIndices[element]), elementFPs[element], 1).repeat(1, 1, 3)
            self.fp_imageIndices[element] = torch.tensor(self.fp_imageIndices[element], dtype=torch.int64)
            self.force_AtomIndices[element] = torch.tensor(self.force_AtomIndices[element], dtype=torch.int64)

            # For pyamff calculator
            if not batch:
                if forceIndices is None:
                    # forceIndices = torch.tensor(self.force_AtomIndices[element])
                    forceIndices = self.force_AtomIndices[element]
                else:
                    forceIndices = torch.cat([forceIndices, torch.tensor(self.force_AtomIndices[element])])

        if not batch:
            self.force_AtomIndices = forceIndices

        self.natomsPerImageForce = torch.tensor(natomsPerImage, dtype=torch.int64)
        self.natomsPerImageEnergy = torch.tensor(natoms, dtype=torch.int64)
        self.ntotalAtoms = ntotalAtoms

    def sort_fortranFPprimes(self, keylist=None, fortran_fpprimesDB=None):
        # Fortran dgdx
        self.fortran_dgdx = {}
        if fortran_fpprimesDB is not None:
            for k in range(len(keylist)):
                self.fortran_dgdx[keylist[k]] = fortran_fpprimesDB[keylist[k]]

    def normalize_fortrandgdx(self, fprange, magnitudeScale):
        elements = list(fprange.keys())
        for element in elements:
            elemidx = elements.index(element)
            total_atoms = len(self.allElement_fps[element])
            mags = magnitudeScale[element].repeat(total_atoms,1)
            #print ('mags')
            #print (mags)
            ### Not sure if this is efficient
            len_mags = magnitudeScale[element].shape[0]
            fortran_mags = magnitudeScale[element].numpy().reshape(len_mags,1)
            fortran_mags = np.repeat(fortran_mags,3,axis=1)
            fortran_mags = np.repeat(fortran_mags[:,:,np.newaxis],total_atoms,axis=2)
            fortran_mags = np.repeat(fortran_mags[:,:,:,np.newaxis],total_atoms,axis=3)
            self.fortran_dgdx[elemidx] = np.multiply(self.fortran_dgdx[elemidx],fortran_mags)

    def sortFPsList(self, atomSymbols, fpdata, elementFPs,
                    properties=None, fpDer=None, fortran_fpprimesDB=None, batch=True):
        """
        This function generates the inputs to the tensorflow graph for the selected images.
        The essential problem is that each neural network is associated with a specific element type.
        Thus, atoms in each ASE image need to be sent to different networks.

        Inputs:
        atomSymbols: a list of chemical symbols
        fpdata: a list of fingerprint for each atom, 
                [[fp_atom1], [fp_atom2], ..., [fp_atomN]]
        elementFPs: a Ordered dictionary of number of fingerprints for each type of element (e.g. {'C':2,'O':5}, etc)
        keylist: a list of hashs into the fingerprintDB that we want to create inputs for
        fingerprintDerDB: a database of fingerprint derivatives, as taken from the descriptor

        Outputs:
        allElement_fps: a dictionary of fingerprint inputs to each element's neural
            network
            Note: G_H_1 represent the 1st fingerprint centered on 'H'
                  Assume we have g1 fingerprints that are centered on 'H',
                                 g2 fingerprints that are centered on 'Pd'
            {'H':tensor([ [G_H_1,G_H_2,...,G_H_g1],  Atom 1  in Image 1
                          [G_H_1,G_H_2,...,G_H_g1],  Atom 2  in Image 1
                                 ...
                          [G_H_1,G_H_2,...,G_H_g1],  Atom N1 in Image 1

                          [G_H_1,G_H_2,...,G_H_g1],  Atom 1  in Image 2
                                 ...
                          [G_H_1,G_H_2,...,G_H_g1],  Atom N2 in Image 2
                                 ...
                          [G_H_1,G_H_2,...,G_H_g1],  Atom NM in Image M
                         ])
             'Pd':tensor([[G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom 1  in Image 1
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom 2  in Image 1
                                 ...
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom N1 in Image 1

                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom 1  in Image 2
                                 ...
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom N2 in Image 2
                                 ...
                          [G_Pd_1,G_Pd_2,...,G_Pd_g2],  Atom NM in Image M
                         ])
              }
        ntotalAtoms: the total number of atoms in the training batch

        dgdx: dictionary of fingerprint derivatives. Grouped by elements:          contrib. From      contrib. TO
             {'H': ([  [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Ggx, Ggy, Ggz]],        neighbor 1           atom 1
                         -----------     ------------       --------------
                            1st fp         2nd fp              gth fp
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor 2           atom 1
                                               ...                                     ...
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor M1          atom 1
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor 1           atom 2
                                               ...                                     ...
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor M2          atom 2
                                                .                                       .                  .
                                                .                                       .                  .
                                                .                                       .                  .

                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor 1           atom N
                                               ...                                     ...
                       [[G1x, G1y, G1z],[G2x, G2y, G2z]...,[Gmx, Gmy, Gmz]]         neighbor MN          atom N
                           (Note: N is total number of 'element' atoms in training images)

                               ] ),
                            'Pd': ()
                           },
             }
        """
        elements = elementFPs.keys()
        #self.nimages = len(keylist)
        self.nimages = 1
        self.allElement_fps = {}
        allAtomIndices = [] # store the location of atom in the whole tensor
        allfpIndices = []   # store index of each fingerpint in allElement_fps
        natoms = []
        natomsPerImage = []
        self.fprange = {}
        self.fp_imageIndices = {}
        for element in elements:
            self.allElement_fps[element] = []
            self.fp_imageIndices[element] = []
        #print(allElement_fps.keys())
        tlocation = 0
        # For pyamff calculator
        if properties is not None:
            self.energies.extend(properties[0])
            self.forces.extend(properties[1])

        #fp = fingerprintDB[keylist[j]]
        #atomSymbols, fpdata = zip(*fp)
        nElement = {}
        allfpIndices = []
        allAtomIndices = []
        natom = len(atomSymbols)
        j=0
        for i in range(len(atomSymbols)):
            self.allElement_fps[atomSymbols[i]].append(list(fpdata[i][:elementFPs[atomSymbols[i]]]))
            currlocation = len(self.allElement_fps[atomSymbols[i]])
            allAtomIndices.append(tlocation)
            allfpIndices.append([index for index in range( \
                                  (currlocation-1) * elementFPs[atomSymbols[i]], \
                                   currlocation * elementFPs[atomSymbols[i]])
                                 ])
            self.fp_imageIndices[atomSymbols[i]].append([j])
            tlocation += 1
            if atomSymbols[i] not in nElement:
                nElement[atomSymbols[i]] = 1
            else:
                nElement[atomSymbols[i]] += 1
            natomsPerImage.append(natom)
        natoms=natom

        #self.energies = torch.tensor(self.energies,dtype=torch.double)
        #self.forces = torch.tensor(self.forces,dtype=torch.double)

        self.natomsPerElement = {}
        ntotalAtoms = tlocation
        for element in elements:
        #    minv = np.min(self.allElement_fps[element], axis=0)
        #    maxv = np.max(self.allElement_fps[element], axis=0)
        #    self.fprange[element] = [minv, maxv, maxv-minv]
            self.natomsPerElement[element] = len(self.allElement_fps[element])
        # Set up the array for atom-based fingerprint derivatives
        self.dgdx = {}
        dgdx_XYZindices = {}
        self.force_AtomIndices = {}  # Used to sum forces over atoms
        self.dEdg_AtomIndices = {}   # Used to fetch dEdg to be used to multiply with dgdx tensor
        dict_init = {}
        if fpDer is not None:
            for element in elements:
                self.dgdx[element] = []
                self.force_AtomIndices[element] = []
                dict_init[element] = []
                self.dEdg_AtomIndices[element] = []

            #fp = fingerprintDB[keylist[j]]
            #atomSymbols, fpdata = zip(*fp) #Fetch atomSymbols for each image

            #fpDer = fingerprintDerDB[keylist[j]]

            # Iterate over all atoms in the image
            #natom = natoms[keylist[j]]
            dgdx_image = copy.deepcopy(dict_init)
            for wrtIndex in range(natoms):     # Make sure images are ordered by element and elementFPs
                wrtSymbol = atomSymbols[wrtIndex]
                # TODO: iterate over neighborlist only
                for centerIndex in range(natoms):
                    dgdx_temp = copy.deepcopy(dict_init)
                    centerSymbol = atomSymbols[centerIndex]
                    for direction in range(3):
                        try:
                            dgdx_temp[centerSymbol].append(fpDer[(wrtIndex, wrtSymbol,centerIndex, centerSymbol, direction)])
                        except:
                            pass
                    # TODO: there may exist an atom with 0 neighbors
                    if len(dgdx_temp[centerSymbol]) > 1:
                        self.dgdx[centerSymbol].append(np.array(dgdx_temp[centerSymbol]).T.tolist())
                        self.dEdg_AtomIndices[centerSymbol].append(allfpIndices[centerIndex])
                        self.force_AtomIndices[centerSymbol].append([allAtomIndices[wrtIndex]]*3)
            forceIndices = None
            for element in elements:
                self.allElement_fps[element] = np.array(self.allElement_fps[element])
                self.dgdx[element] = np.array(self.dgdx[element])
                self.dEdg_AtomIndices[element] = np.repeat(np.reshape(np.array(self.dEdg_AtomIndices[element]),\
                    (len(self.dEdg_AtomIndices[element]), elementFPs[element], 1)), 3, axis=2)
                self.fp_imageIndices[element] = np.array(self.fp_imageIndices[element])
                self.force_AtomIndices[element] = np.array(self.force_AtomIndices[element])
                # For pyamff calculator
                if not batch:
                    if forceIndices is None:
                        #forceIndices = torch.tensor(self.force_AtomIndices[element])
                        forceIndices = self.force_AtomIndices[element]
                    else:
                        forceIndices = [*forceIndices, *self.force_AtomIndices[element]]
            if not batch:
                self.force_AtomIndices = forceIndices
        self.natomsPerImageForce = natomsPerImage
        self.natomsPerImageEnergy = [natoms]
        self.ntotalAtoms = ntotalAtoms
        #return (allElement_fps, fp_imageIndices, fprange, dgdx, dEdg_AtomIndices, force_AtomIndices,
        #        natomsPerElement, torch.tensor(natomsPerImage), torch.tensor(natoms), ntotalAtoms)


    # def sortFPs_calc(self, atomSymbols, fingerprintDB, elementFPs, fingerprintDerDB=None):
    #     elements = elementFPs.keys()
    #     self.nimages = 1  # Since keylist is always [0], we only have one image
    #     self.allElement_fps = {}
    #     allAtomIndices = []  # store the location of atom in the whole tensor
    #     allfpIndices = []  # store index of each fingerprint in allElement_fps
    #     natomsPerImage = []
    #     self.fprange = {}
    #     self.fp_imageIndices = {}
        
    #     for element in elements:
    #         self.allElement_fps[element] = []
    #         self.fp_imageIndices[element] = []

    #     tlocation = 0
    #     # fpdata = fingerprintDB
    #     natom = len(atomSymbols)
    #     for i in range(natom):
    #         self.allElement_fps[atomSymbols[i]].append(fingerprintDB[i][:elementFPs[atomSymbols[i]]])
    #         currlocation = len(self.allElement_fps[atomSymbols[i]])
    #         allAtomIndices.append(tlocation)
    #         allfpIndices.append([index for index in range((currlocation - 1) * elementFPs[atomSymbols[i]],currlocation * elementFPs[atomSymbols[i]])])
    #         self.fp_imageIndices[atomSymbols[i]].append([0])  # Single image index
    #         tlocation += 1
    #         natomsPerImage.append(natom)

    #     self.natomsPerElement = {}
    #     ntotalAtoms = tlocation
    #     for element in elements:
    #         self.natomsPerElement[element] = len(self.allElement_fps[element])
    #         if len(self.allElement_fps[element]) > 0:
    #             minv = np.min(self.allElement_fps[element], axis=0)
    #             maxv = np.max(self.allElement_fps[element], axis=0)
    #             self.fprange[element] = [minv, maxv, maxv - minv]
    #         # print(np.array(self.allElement_fps[element]).shape)

    #     # Set up the array for atom-based fingerprint derivatives
    #     self.dgdx = {}
    #     self.force_AtomIndices = {}  # Used to sum forces over atoms
    #     self.dEdg_AtomIndices = {}  # Used to fetch dEdg to be used to multiply with dgdx tensor

    #     for element in elements:
    #         self.dgdx[element] = []
    #         self.force_AtomIndices[element] = []
    #         self.dEdg_AtomIndices[element] = []

    #     '''----------optimized part---------------'''
    #     for wrtIndex in range(natom):  # make sure images are ordered by element and elementFPs
    #         for centerIndex in range(natom):
    #             centerSymbol = atomSymbols[centerIndex]
    #             key = (wrtIndex, atomSymbols[wrtIndex], centerIndex, atomSymbols[centerIndex], 0) # if key with direction = 0 exists, it also exist for direction = 1 and 2
    #             if key in fingerprintDerDB.keys():
    #                 dgdx_temp = np.empty((3, elementFPs[centerSymbol]))
    #                 dgdx_temp[0, :] = fingerprintDerDB[(wrtIndex, atomSymbols[wrtIndex], centerIndex, atomSymbols[centerIndex],0)]
    #                 dgdx_temp[1, :] = fingerprintDerDB[(wrtIndex, atomSymbols[wrtIndex], centerIndex, atomSymbols[centerIndex],1)]
    #                 dgdx_temp[2, :] = fingerprintDerDB[(wrtIndex, atomSymbols[wrtIndex], centerIndex, atomSymbols[centerIndex],2)]

    #                 self.dgdx[centerSymbol].append(torch.tensor(dgdx_temp).T)
    #                 self.dEdg_AtomIndices[centerSymbol].append(allfpIndices[centerIndex])
    #                 self.force_AtomIndices[centerSymbol].append([allAtomIndices[wrtIndex]] * 3)
    #     '''-------------------------'''

    #     for element in elements:
    #         self.allElement_fps[element] = torch.tensor(self.allElement_fps[element], dtype=torch.float32)
    #         # self.dgdx[element] = torch.stack(self.dgdx[element]) if self.dgdx[element] else torch.empty(0, dtype=torch.float32)
    #         self.dgdx[element] = torch.stack(self.dgdx[element])
    #         # print(element, self.dgdx[element].shape)
    #         self.dEdg_AtomIndices[element] = torch.tensor(self.dEdg_AtomIndices[element], dtype=torch.int64).view(len(self.dEdg_AtomIndices[element]), elementFPs[element], 1).repeat(1, 1, 3)
    #         self.fp_imageIndices[element] = torch.tensor(self.fp_imageIndices[element], dtype=torch.int64)
    #         self.force_AtomIndices[element] = torch.tensor(self.force_AtomIndices[element], dtype=torch.int64)

    #     self.natomsPerImageForce = torch.tensor(natomsPerImage, dtype=torch.int64)
    #     self.natomsPerImageEnergy = torch.tensor([natom], dtype=torch.int64)  # Single image
    #     self.ntotalAtoms = ntotalAtoms
    #     # return (allElement_fps, fp_imageIndices, fprange, dgdx, dEdg_AtomIndices, force_AtomIndices,
    #     #        natomsPerElement, torch.tensor(natomsPerImage), torch.tensor(natoms), ntotalAtoms)
    
    def stackFPs_calc(self, acf):

        self.elements = list(acf.allElement_fps.keys())
        elements = list(acf.allElement_fps.keys())
        
        self.allElement_fps = acf.allElement_fps
        self.fp_imageIndices = acf.fp_imageIndices
        self.dgdx = acf.dgdx
        self.dEdg_AtomIndices = acf.dEdg_AtomIndices
        self.force_AtomIndices = acf.force_AtomIndices
        self.natomsPerElement = acf.natomsPerElement
        self.natomsPerImageForce = acf.natomsPerImageForce
        self.natomsPerImageEnergy = acf.natomsPerImageEnergy
        self.ntotalAtoms = acf.ntotalAtoms
        self.nimages = acf.nimages
        self.energies = acf.energies
        self.forces = acf.forces
        self.fortran_dgdx = acf.fortran_dgdx

        forceIndices = []
        for element in elements:
            forceIndices.append(acf.force_AtomIndices[element])
        
        self.force_AtomIndices = torch.cat(forceIndices)

        self.natomsPerImageFxyz = torch.reshape(self.natomsPerImageForce, (self.ntotalAtoms, 1)).repeat(1, 3)
