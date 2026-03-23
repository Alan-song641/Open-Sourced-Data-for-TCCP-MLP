import numpy as np
import pickle
import math
import os, sys
import time
import torch
from ase.formula import Formula
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
from pyamff.utilities import fileIO as io
from pyamff.utilities.preprocessor import normalizeParas
from collections import OrderedDict
import itertools
import tempfile
import copy
import torch.distributed as dist

from pyamff.fingerprints.ewald_FPs import Ewald_FPs
import bz2file as bz2
try:
    from pyamff import fmodules
    FMODULES = True
except:
    FMODULES = False

class Fingerprints():

    """
    An implementation of the Behler-Parrinello descriptors.

    References
    ----------
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401.

    """


    def __init__(self, uniq_elements, filename='fpParas.dat', nfps=None, active=False):
        self.filename = filename
        self.nfps = nfps
        self.uniq_elements = uniq_elements
        #print('uniq', uniq_elements)
        self.nelement = len(self.uniq_elements)
        self.max_nfps = max(nfps.values())
        self.forceEngine = 0
        self.max_neighs = 100
        nelement = len(nfps.keys())
        self.coef =  np.zeros(nelement, order='F')
        self.minFPs = {}
        self.maxFPs = {}
        self.fprange = {}
        if active == False:
            self.coef = fmodules.fpcalc.read_fpparas(self.filename, nelement)

    def toIndex(self, symbols, nAtoms):
        elementDict = dict(zip(self.uniq_elements,range(1, self.nelement+1)))
        nsymbols = np.zeros(nAtoms,dtype=np.dtype('i4'))
        for i in range(nAtoms):
            try:
                nsymbols[i] = elementDict[symbols[i]]
            except:
                sys.stderr.write('Element %s has no fingerprints defined' % (symbols[i]))
                sys.exit(2)
        return nsymbols

    def calcFPs(self, atoms, chemsymbols, if_force=True):
        fingerprints = []
        fingerprintprimes = {}
        nAtoms = len(atoms)
        symbols = self.toIndex(chemsymbols, nAtoms)
        pos_car = atoms.get_positions()
        # cell = atoms.cell.array.astype(np.double, order='F')
        cell = atoms.cell.array
        fps =  np.zeros([nAtoms, self.max_nfps], order='F')
        dfps = np.zeros([nAtoms, self.max_neighs, 3, self.max_nfps], order='F')
        neighs = np.zeros([nAtoms, self.max_neighs], order='F')
        num_neigh = np.zeros(nAtoms, dtype=np.dtype('i4'))
        nneigh = np.zeros(nAtoms, dtype=np.dtype('i4'))
        fmodules.fpcalc.max_fps = self.max_nfps
        num_neigh, neighs, nneigh = fmodules.fpcalc.calcfps(pos_car, cell, symbols, len(self.uniq_elements), self.forceEngine)
        #print('fps')
        fps = fmodules.atomsprop.fps
        dfps = fmodules.atomsprop.dfps
        
        fmodules.atomsprop.deallocate_outputs
        
        if if_force:
            for i in range(0, nAtoms):
                wrtsymbol = chemsymbols[i]
                #fingerprints.append((chemsymbols[i], fps[i][:self.nfps[wrtsymbol]]))
                for j in range(0,3):
                    for k in range(0, num_neigh[i]+1):
                        if k == 0:
                            centerIndex  = i
                            centersymbol = wrtsymbol
                            #fingerprintprimes[(i, chemsymbols[i], i, chemsymbols[i], j)] = dfps[i,k,j,:]
                        else:
                            if neighs[i, k-1] <= nAtoms:
                                centerIndex = neighs[i, k-1] - 1
                                centersymbol = chemsymbols[centerIndex]
                            else:
                                continue
                        if (i, wrtsymbol, centerIndex, centersymbol, j) in fingerprintprimes.keys():
                            fingerprintprimes[(i, wrtsymbol, centerIndex, centersymbol, j)] += dfps[i,k,j,:self.nfps[centersymbol]]
                        else:
                            fingerprintprimes[(i, wrtsymbol, centerIndex, centersymbol, j)] = dfps[i,k,j,:self.nfps[centersymbol]]
        #print('fps done')
        fmodules.fpcalc.atomscleanup()
        return fps, fingerprintprimes

    #from memory_profiler import profile
    #@profile
    def loop_images(self, rank, size, nFPs, num_batch, batchIDs, 
                    trainingimages, properties, normalize, logger, 
                    fpsdir=None, useexisting=False, test=False, chg_grad=False,
                    real_space_cut=0.0, total_ion_charge=True):
                    
        #import tracemalloc 
        #tracemalloc.start()
        #snap1 = tracemalloc.take_snapshot()
        fpDb = {}
        fpDerDb = {}
        fpData = OrderedDict() 
        # Alan: create Ewald_FPs class
        coulData = OrderedDict()

        fpData_temp = {}
        keybatch = []
        #for ele in nFPs.keys():
        #    aEfps[ele] = []
        fptime = 0
        #for struct in trainingimages.keys():
        for struct in batchIDs:
            if logger and struct % 20 == 0:
                logger.info('  Calculating FPs for image %d', struct)
            # Get FPs and FPprimes for each structure
            st = time.time()
            if not useexisting:
                chemsymbols = trainingimages[struct].get_chemical_symbols()
                fingerprints, fingerprintprimes = self.calcFPs(trainingimages[struct], chemsymbols)
                et = time.time()
                fptime += et-st
                acf = atomCenteredFPs()
                # Alan: create Ewald_FPs class
                efp = Ewald_FPs(structure=trainingimages[struct], 
                                chg_grad=chg_grad,
                                real_space_cut=real_space_cut,
                                compute_forces=True,
                                total_ion_charge=total_ion_charge)

                # For pyamff calculator
                if properties is None:
                    p1 = None
                    acf.sortFPs(fpDb, nFPs, p1, [struct], fpDerDb, batch=False)
                    return acf

                # Store FPs and FPprimes as acf objects and make readable by pytorch/fortran machine learning
                #acf.sortFPs(fpDb, nFPs, p1, [struct], fpDerDb)
                acf.sortFPs(chemsymbols, fingerprints, nFPs, properties, [struct], fingerprintprimes)
                #acf.sortFPsList(chemsymbols, fingerprints, nFPs, properties[struct], fingerprintprimes)

                fpData[struct] = acf 
                #print(acf.allElement_fps)
                if struct%num_batch == 0 or struct == len(trainingimages)-1:
                    for key in fpData.keys():
                        f_name = fpsdir+'/fps_{}.pckl'.format(key)
                        with bz2.BZ2File(f_name, 'wb') as f:
                            pickle.dump(fpData[key],f)
                    fpData = {}
                
                coulData[struct] = efp
                if struct % num_batch == 0 or struct == len(trainingimages) - 1:
                    for key in coulData.keys():
                        f_name = fpsdir + '/efps_{}.pckl'.format(key)
                        with bz2.BZ2File(f_name, 'wb') as f:
                            pickle.dump(coulData[key], f)
                    coulData = {}

            else:
                #print('load existing')
                fname = os.path.join(fpsdir, 'fps_{}.pckl'.format(struct))
                with bz2.BZ2File(fname, 'rb') as f1:
                    acf = pickle.load(f1)
                cname = os.path.join(fpsdir, 'efps_{}.pckl'.format(struct))
                with bz2.BZ2File(cname, 'rb') as f2:
                    efp = pickle.load(f2)
            
            if normalize:
                for k, v in acf.allElement_fps.items():
                    if len(v) == 0:
                        print('Warning: no atoms of element', k, 'in image', struct)
                        self.minFPs[k] = torch.full((self.nfps[k],), 1e10, dtype=torch.float64)
                        self.maxFPs[k] = torch.full((self.nfps[k],), -1e10, dtype=torch.float64)
                        continue

                    if k not in self.minFPs:
                        #minv[k] = np.amin(v, axis=0)
                        #maxv[k] = np.amax(v, axis=0)
                        self.minFPs[k] = torch.amin(v, dim=0)
                        self.maxFPs[k] = torch.amax(v, dim=0)
                        
                    else:
                        #minv[k] = np.minimum(minv[k], np.amin(v, axis=0))
                        #maxv[k] = np.maximum(maxv[k], np.amax(v, axis=0))
                        self.minFPs[k] = torch.minimum(self.minFPs[k], torch.amin(v, dim=0))
                        self.maxFPs[k] = torch.maximum(self.maxFPs[k], torch.amax(v, dim=0))
        
        for ele in acf.allElement_fps.keys():
          dist.all_reduce(self.minFPs[ele], op=dist.ReduceOp.MIN)
          dist.all_reduce(self.maxFPs[ele], op=dist.ReduceOp.MAX)
          self.fprange[ele] = [self.minFPs[ele], self.maxFPs[ele], self.maxFPs[ele]-self.minFPs[ele]]

        if rank == 0 and test == False: # in test case, I can let normalize happen but wont let the code dump it. 
           f_name = fpsdir+'/fprange.pckl'
           with bz2.BZ2File(f_name, 'wb') as f:
               pickle.dump(self.fprange,f)
           #print("          Fingerprints done, time: %.2f s" % fptime)
        #if normalize:
        #    for ele in nFPs.keys():
        #        self.fprange[ele] = [minv[ele], maxv[ele], maxv[ele]-minv[ele]]
            #fprange, magnitudeScale, interceptScale = normalizeParas(fprange)

        #first_size, first_peak = tracemalloc.get_traced_memory()
        #print('loopimages:',first_peak/1024/1024)

        #save fp dictionary with fprange and fpdata to pickle file. 'fps.pckl' is default file name
        #io.save_data(fp, fpfilename)
        fmodules.fpcalc.cleanup()
        #return fprange, magnitudeScale, interceptScale

