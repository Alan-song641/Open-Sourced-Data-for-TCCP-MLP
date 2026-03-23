#from . import generateTensorFlowArrays
#from .. import FileDatabase
import numpy as np
import torch
import copy, math, sys
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
from pyamff.mlModels.lossFunctions import calc_mse
from collections import OrderedDict, Counter

#def fetchProp(images, startkey=0, forceTraining=False):

class Scaler:
    def __init__(self, scalerType='LinearScaler',forceTraining=True, activeLearning=False, loss_type=None, cohe=False, device=torch.device("cpu")):
        self.scalerType = scalerType
        #print('self.scalerType', self.scalerType)
        self.activeLearning = activeLearning
        self.forceTraining = forceTraining
        self.loss_type = loss_type
        self.cohe = cohe
        self.device = device

    def set_scaler(self):
        if self.scalerType == 'NoScaler':
            return NonScaler(self.scalerType, self.forceTraining, self.activeLearning, self.loss_type, self.cohe, self.device)
        if self.scalerType == 'LinearScaler':
            return LinearScaler(self.scalerType, self.forceTraining, self.activeLearning, self.loss_type, self.cohe, self.device)
        if self.scalerType == 'MinMaxScaler':
            return MinMaxScaler(self.scalerType, self.forceTraining, self.activeLearning, self.loss_type, self.cohe, self.device)
        if self.scalerType == 'LogScaler':
            return LogScaler(self.scalerType, self.forceTraining, self.activeLearning, self.loss_type, self.cohe, self.device)
        if self.scalerType == 'STDScaler':
            return STDScaler(self.scalerType, self.forceTraining, self.activeLearning, self.loss_type, self.cohe, self.device)
        return None

class LinearScaler(Scaler):

    def __init__(self, scalerType='LinearScaler', forceTraining=True, activeLearning=False, loss_type=None, cohe=False,device=torch.device("cpu")):
        super(LinearScaler, self).__init__(scalerType, forceTraining, activeLearning, loss_type, cohe)
        self.slope = None
        self.intercept = None
        self.adjust = False
        self.fc_slope = None
        self.fc_intercept = None
        self.device = device

    def scaling(self, energies, forces, natoms):
        if len(energies) >1 and not self.activeLearning:
            self.intercept = np.mean(energies)
            n_energies = energies - self.intercept
            self.slope = np.mean(np.abs(n_energies))
        else:
            self.intercept = 0.5
            n_energies = energies - self.intercept
            self.slope = (0.5/(np.abs(n_energies)))[0]
        if not self.adjust:
            energies = n_energies
        return energies, forces

    def calculate_mse(self, predEs, predFs, targetEs, targetFs, natomsEnergy, natomsForce):
        natomsEnergy = natomsEnergy.to(self.device)
        natomsForce = natomsForce.to(self.device)
        targetEs = targetEs.to(self.device)
        targetFs = targetFs.to(self.device)
        if self.loss_type != 'SE':
            energyMSE, forceMSE = calc_mse(predEs, predFs, targetEs, targetFs,
                                           natomsEnergy,
                                           natomsForce)
            return energyMSE, forceMSE
        else:
            return 0.0, 0.0

class NonScaler(Scaler):

    def __init__(self, scalerType='NoScaler', forceTraining=True, activeLearning=False, loss_type=None, cohe=False,device=torch.device("cpu")):
        super(NonScaler, self).__init__(scalerType, forceTraining, activeLearning, loss_type, cohe)
        self.slope = None
        self.intercept = None
        self.adjust = None
        self.fc_slope = None
        self.fc_intercept = None
        self.device = device

    def scaling(self, energies, forces, natoms):
        self.intercept = 0.0
        n_energies = energies
        self.slope = 1.0
        if self.adjust:
           import warnings
           warnings.warn('adjust option is not applicable to non-scaler method, check Linear, STD or MixMax')
        return energies, forces

    def calculate_mse(self, predEs, predFs, targetEs, targetFs, natomsEnergy, natomsForce):
        natomsEnergy = natomsEnergy.to(self.device)
        natomsForce = natomsForce.to(self.device)
        targetEs = targetEs.to(self.device)
        targetFs = targetFs.to(self.device)
        if self.loss_type != 'SE':
            energyMSE, forceMSE = calc_mse(predEs, predFs, targetEs, targetFs,
                                           natomsEnergy,
                                           natomsForce)
            return energyMSE, forceMSE
        else:
            return 0.0, 0.0

class MinMaxScaler(Scaler):
    def __init__(self, scalerType='MinMaxScaler', forceTraining=True, activeLearning=False, loss_type=None, cohe=False, device=torch.device("cpu")): 
      super(MinMaxScaler, self).__init__(scalerType, forceTraining, activeLearning, loss_type, cohe)
      self.eMinMax = None
      self.fMinMax = None
      self.eRange = None
      self.slope = None
      self.intercept = None
      self.fc_slope = None
      self.fc_intercept = None
      self.adjust = None

    def scaling(self, energies, forces, natoms):
      minE = np.amin(energies)
      maxE = np.amax(energies)
      self.intercept = minE
      self.slope =  maxE-minE
      if self.adjust:
          return energies, forces
   
      energies =(energies - minE)/(maxE-minE)
      if self.forceTraining: 
        fc_arr = []
        for fc in forces:
          fc_arr.extend(fc)
        minF = np.amin(fc_arr)
        maxF = np.amax(fc_arr)
        self.fc_intercept = minF
        self.fc_slope = maxF - minF
        i = 0
        for fc in forces:
          forces[i] = ((np.array(forces[i]) - minF)/self.fc_slope).tolist()
          i+=1
      eout = open('cohe.dat','w')
      i = 0
      for e in energies:
        eout.write("{:6d}  {:.6f}\n".format(i, e))
        i+=1
      eout.close() 
      print('scaler', self.fc_intercept)
      #self.eMinMax = np.array([minE, maxE])
      #self.fMinMax = np.array([minF, maxF])
      #self.eRange = self.eMinMax[1] - self.eMinMax[0]
      return energies, forces

    def calculate_mse(self, predEs, predFs, targetEs, targetFs, natomsEnergy, natomsForce):
      natomsEnergy = natomsEnergy.to(self.device)
      natomsForce = natomsForce.to(self.device)
      targetEs = targetEs.to(self.device)
      targetFs = targetFs.to(self.device) 

      energyMSE, forceMSE = calc_mse(predEs, predFs, targetEs, targetFs,
                                     natomsEnergy,
                                     natomsForce)
      if not self.adjust:
        energyMSE = energyMSE * self.slope * self.slope
        forceMSE = forceMSE * self.fc_slope * self.fc_slope
      return energyMSE, forceMSE

class LogScaler(Scaler):
    def __init__(self, scalerType='LogScaler', forceTraining=True, activeLearning=False, loss_type=None, cohe=False, device=torch.device("cpu")): 
      super(LogScaler, self).__init__(scalerType, forceTraining, activeLearning, loss_type, cohe)
      #print('Log self.scalerType', self.scalerType)
      self.eMinMax = None
      self.eRange = None
      self.fMinMax = None
      self.fRange = None

    def scaling(self, energies, forces, natoms):
      minE = np.amin(energies)
      maxE = np.amax(energies)
      self.eRange = maxE - minE
      energies = np.log((energies - minE)/(self.eRange)+1)
      if self.forceTraining: 
          minF = np.amin(forces)
          maxF = np.amax(forces)
          if isinstance(minF, list):
             minF = np.amin(minF)
             maxF = np.amax(maxF)
             self.fRange = maxF - minF
             scaled_forces = []
             for fs in forces:
                scaled_forces.append(np.log((np.array(fs) - minF)/(self.fRange)+1).tolist())
          else:
             self.fRange = maxF - minF
             print('forces:', type(forces))
             scaled_forces = np.log(((forces - minF)/(self.fRange)+1.0).astype('float')).tolist()
          #forces = forces.tolist()
      self.eMinMax = np.array([minE, maxE])
      self.fMinMax = np.array([minF, maxF])
      return energies, scaled_forces

    def calculate_mse(self, predEs, predFs, targetEs, targetFs, natomsEnergy, natomsForce):
      natomsEnergy = natomsEnergy.to(self.device)
      natomsForce = natomsForce.to(self.device)
      targetEs = targetEs.to(self.device)
      targetFs = targetFs.to(self.device) 

      predEs = torch.exp(predEs)
      targetEs = torch.exp(targetEs)
      predFs = torch.exp(predFs)
      targetFs = torch.exp(targetFs)
      energyMSE, forceMSE = calc_mse(predEs, predFs, targetEs, targetFs,
                                     natomsEnergy,
                                     natomsForce)
      energyMSE = energyMSE * self.eRange * self.eRange
      return energyMSE, forceMSE

#Ref: N2P2
class STDScaler(Scaler):
    def __init__(self, scalerType='STDScaler', forceTraining=True, activeLearning=False, loss_type=None, cohe=False, device=torch.device("cpu")): 
        super(STDScaler, self).__init__(scalerType, forceTraining, activeLearning, loss_type, cohe)
        self.slope = 1.0 #e_std
        self.intercept = 0.0 #mean e per atom
        self.f_std = 1.0
        self.ef_coef = 1.0

    def scaling(self, energies, forces, natoms):
        if self.cohe:
            self.intercept = np.mean(energies)
            self.slope = np.std(energies, ddof=1)
            energies = (energies - self.intercept)/self.slope
        else:
            e_peratom = energies/natoms
            self.intercept = np.mean(e_peratom)
            self.slope = np.std(e_peratom, ddof=1)
            energies = (energies - natoms*self.intercept)/self.slope
        fc_sum = 0.0
        tsize = 0
        for fc in forces:
            fc_sum += np.sum(np.square(fc))
            tsize += np.size(fc)
        self.f_std = np.sqrt(fc_sum/(tsize-1))  #Assume forces mean is zero
        i = 0
        for fc in forces:
            forces[i] = fc / self.f_std
            i+=1
        self.ef_coef = self.slope/self.f_std
        print('std', self.slope, self.intercept)
        return energies, forces

    def calculate_mse(self, predEs, predFs, targetEs, targetFs, natomsEnergy, natomsForce):
      natomsEnergy = natomsEnergy.to(self.device)
      natomsForce = natomsForce.to(self.device)
      targetEs = targetEs.to(self.device)
      targetFs = targetFs.to(self.device)

      energyMSE, forceMSE = calc_mse(predEs, predFs, targetEs, targetFs,
                                     natomsEnergy,
                                     natomsForce)
      energyMSE = energyMSE * self.slope * self.slope
      forceMSE = forceMSE * self.f_std * self.f_std
      return energyMSE, forceMSE

def fetchProp(images, refEs=None,scaler=None, forceTraining=False, activeLearning=False):
    energies = []
    forces = []
    #forces_arr = []
    natoms = []
    properror = False
    if refEs is None:
        for i in range(len(images)):
            nsymbols = Counter(list(images[i].symbols))
            try:
                energies.append(images[i].get_potential_energy(apply_constraint=False))
                natoms.append(len(images[i]))
            except:
                sys.stderr.write('Image %d has no property of energy\n'%(i))
                properror = True
                pass
            if forceTraining:
                try:
                    forces.append(images[i].get_forces(apply_constraint=False).tolist())
                except:
                    sys.stderr.write('Image %d has no property of force\n'%(i))
                    properror = True
                    pass
    else:
      print('Using cohensive energy')
      eout = open('cohe.dat','w')
      for i in range(len(images)):
          nsymbols = Counter(list(images[i].symbols))
          nelems = []
          res = []
          for key in refEs:
              if key not in nsymbols:
                  continue
              nelems.append(nsymbols[key])
              res.append(refEs[key])
          try:
              temp_e = (images[i].get_potential_energy(apply_constraint=False) \
                           - np.dot(np.array(nelems), np.array(res)))/len(images[i])
              #temp_e = images[i].get_potential_energy(apply_constraint=False) - np.dot(np.array(nelems), np.array(res))
              natoms.append(len(images[i]))
              eout.write("{:6d}  {:.6f}\n".format(i, temp_e))
              energies.append(temp_e)
              #energies.append((images[i].get_potential_energy(apply_constraint=False)
              #                - np.dot(np.array(nelems), np.array(res))))
          except:
              sys.stderr.write('Image %d has no property of energy\n'%(i))
              properror = True
              pass
          if forceTraining:
              try:
                  forces.append(images[i].get_forces(apply_constraint=False).tolist())
                  #forces_arr.extend(images[i].get_forces(apply_constraint=False).tolist())
              except:
                  sys.stderr.write('Image %d has no property of force\n'%(i))
                  properror = True
                  pass

          eout.flush()
      eout.close()

    if properror:
        sys.exit(2)
    properties = OrderedDict()
    trainingimages = OrderedDict()
    energies = np.array(energies)
    forces = np.array(forces, dtype="object")
    #forces = np.array(forces)
    energies, forces = scaler.scaling(energies, forces, np.array(natoms))
    for i in range(len(images)):
        properties[i] = [np.array([energies[i]]), forces[i]]
        trainingimages[i] = images[i]
    return trainingimages, properties, scaler

#collate_fn: imagesFPs: a list of atomsCenteredFps
#def batchGenerator(acfs):
    #keylist = list(imagesFPs.keys())
    #interval = int(len(keylist)/numbbatch)
    #if len(keylist)%numbbatch != 0:
    #  numbbatch += 1
    #fpRange_temp = batch.fpRange
    #for element in elements:
    #   if element not in minlist:
    #     minlist[element] = np.array([fpRange_temp[element][0].tolist()])
    #     maxlist[element] = np.array([fpRange_temp[element][1].tolist()])
    #   else:
    #     minlist[element] = np.concatenate(( minlist[element], np.array([fpRange_temp[element][0].tolist()])), axis = 0)
    #     maxlist[element] = np.concatenate(( maxlist[element], np.array([fpRange_temp[element][1].tolist()])), axis = 0)
    #for element in elements:
    #  minv = np.min( minlist[element], axis=0)
    #  maxv = np.max( maxlist[element], axis=0)
    #  fpRange[element] = [minv, maxv, maxv-minv]
#    return batch


def batchNormalize(batch, fpRange, magnitudeScale, interceptScale, force_coefficient=None):
    print('Normalizing')
    elements = list(fpRange.keys())
    #print("fpRange", fpRange)
    #for batch in batches:
    for element in elements:
        total_atoms = len(batch.allElement_fps[element])
        mags = magnitudeScale[element].repeat(total_atoms,1)
        inters = interceptScale[element].repeat(total_atoms,1)
        batch.allElement_fps[element] = (torch.mul(torch.tensor(batch.allElement_fps[element]), mags) + inters).double()
        batch.allElement_fps[element].requires_grad = True
        if force_coefficient > 1.e-5:
            batch.dgdx[element] = torch.mul(torch.tensor(batch.dgdx[element]),
                                      torch.flatten(mags)[batch.dEdg_AtomIndices[element]])
            batch.dgdx[element].requires_grad = True
        #batch.allElement_fps[element] = batch.allElement_fps[element].unsqueeze(0)
        #batch.dgdx[element] = batch.dgdx[element].unsqueeze(0)
        #batch.dEdg_AtomIndices[element] = batch.dEdg_AtomIndices[element].unsqueeze(0)
            #dgdx_XYZindices[element][i] = torch.tensor(dgdx_XYZindices[element][i])
            #print('dgdx after:', dgdx[element])
    #print("after normalization",atoms_fps)
    return batch


def normalizeParas(fpRange):
    magnitudeScale = {}
    interceptScale = {}
    for element in fpRange.keys():
        for i in range(len(fpRange[element][0])):
            #avoid overflow
            if fpRange[element][2][i] < 10.**-8:
               fpRange[element][0][i] = -1.
               fpRange[element][2][i] = 2.
        #magnitudeScale[element] = torch.from_numpy(2.0/fpRange[element][2])
        #interceptScale[element] = torch.from_numpy(-2.0 * fpRange[element][0] / fpRange[element][2] - 1 )
        magnitudeScale[element] = 2.0/fpRange[element][2]
        #print ('magnitude scale')
        #print (magnitudeScale[element])
        interceptScale[element] = -2.0 * fpRange[element][0] / fpRange[element][2] - 1
        #print ('intercept scale')
        #print (interceptScale[element])
    return fpRange, magnitudeScale, interceptScale


def normalize(atoms_fps, fpRange, dgdx, dEdg_AtomIndices, forceCoefficient=None):
    normalize_matrix = {}
    magnitudeScale = {}
    interceptScale = {}
    elements = list(atoms_fps.keys())
    for element in elements:
        for i in range(len(fpRange[element][0])):
            #avoid overflow
            if fpRange[element][2][i] < 10.**-8:
                fpRange[element][0][i] = -1.
                fpRange[element][1][i] = 1.
                fpRange[element][2][i] = 2.
        magnitudeScale[element] = torch.from_numpy(2.0/fpRange[element][2])
        interceptScale[element] = torch.from_numpy(-2.0 * fpRange[element][0] / fpRange[element][2] - 1 )

    for element in elements:
        total_atoms = len(atoms_fps[element])
        mags = magnitudeScale[element].repeat(total_atoms,1)
        inters = interceptScale[element].repeat(total_atoms,1)
        atoms_fps[element] = (torch.mul(torch.tensor(atoms_fps[element]), mags) + inters).double()
        atoms_fps[element].requires_grad = True
        if force_coefficient > 1.e-5:
            dgdx[element] = torch.mul(torch.tensor(dgdx[element]),
                                      torch.flatten(mags)[dEdg_AtomIndices[element]])
            dgdx[element].requires_grad = True
            #dgdx_XYZindices[element][i] = torch.tensor(dgdx_XYZindices[element][i])
            #print('dgdx after:', dgdx[element])
    #print("after normalization",atoms_fps)
    return (atoms_fps, dgdx)


def generateInputs(fingerprintDB, elementFPs, 
                      keylist,
                     fingerprintDerDB=None):
      """
      This function generates the inputs to the tensorflow graph for the selected
      images.
      The essential problem is that each neural network is associated with a
      specific element type. Thus, atoms in each ASE image need to be sent to
      different networks.

      Inputs:

      fingerprintDB: a database of fingerprints, as taken from the descriptor

      elementFPs: a Ordered dictionary of number of fingerprints for each type of element (e.g. {'C':2,'O':5}, etc)

      keylist: a list of hashs into the fingerprintDB that we want to create
               inputs for

      fingerprintDerDB: a database of fingerprint derivatives, as taken from the
                        descriptor

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
      allElement_fps = {}
      allAtomIndices = {} #store the location of atom in the whole tensor
      allfpIndices = {}   #store index of each fingerpint in allElement_fps
      natoms = []
      natomsPerImage = []
      fpRange = {}
      fp_imageIndices = {}
      for element in elements:
          allElement_fps[element] = []
          fp_imageIndices[element] = []
      #print(allElement_fps.keys())
      tlocation = 0
      for j in range(len(keylist)):

          fp = fingerprintDB[keylist[j]]
          atomSymbols, fpdata = zip(*fp)
          nElement = {}
          allfpIndices[keylist[j]] = []
          allAtomIndices[keylist[j]] = []
          natom = len(atomSymbols)
          for i in range(len(atomSymbols)):
              allElement_fps[atomSymbols[i]].append(fpdata[i])
              currlocation = len(allElement_fps[atomSymbols[i]])
              allAtomIndices[keylist[j]].append(tlocation)
              allfpIndices[keylist[j]].append([index for index in range( \
                                    (currlocation-1) * elementFPs[atomSymbols[i]], \
                                     currlocation * elementFPs[atomSymbols[i]])
                                   ])
              fp_imageIndices[atomSymbols[i]].append([j])
              tlocation += 1
              if atomSymbols[i] not in nElement:
                  nElement[atomSymbols[i]] = 1
              else:
                  nElement[atomSymbols[i]] += 1
              natomsPerImage.append(natom)
          natoms.append(natom)

      natomsPerElement = {}
      ntotalAtoms = tlocation
      for element in elements:
          minv = np.min(allElement_fps[element], axis=0)
          maxv = np.max(allElement_fps[element], axis=0)
          fpRange[element] = [minv, maxv, maxv-minv]
          print ('fpRange in generate input')
          natomsPerElement[element] = len(allElement_fps[element])
      # Set up the array for atom-based fingerprint derivatives.
      dgdx = {}
      dgdx_XYZindices = {}
      force_AtomIndices = {}   #Used to sum forces over atoms
      dEdg_AtomIndices = {}   #Used to fetch dEdg to be used to multiply with dgdx tensor
      dict_init = {}
      if fingerprintDerDB is not None:
          for element in elements:
              dgdx[element] = []
              force_AtomIndices[element] = []
              dict_init[element] = [] 
              dEdg_AtomIndices[element] = []

          for j in range(len(keylist)):

              fp = fingerprintDB[keylist[j]]
              atomSymbols, fpdata = zip(*fp) #Fetch atomSymbols for each image

              fpDer = fingerprintDerDB[keylist[j]]

              #iterate over all atoms in the image
              #natom = natoms[keylist[j]]
              natom = natoms[j]
              dgdx_image = copy.deepcopy(dict_init)
              for wrtIndex in range(natom):     #make sure images ordered by element and in order of elementFPs
                  wrtSymbol = atomSymbols[wrtIndex]
                  #TODO: iterate over neighborlist only  
                  for centerIndex in range(natom):
                      dgdx_temp = copy.deepcopy(dict_init)
                      centerSymbol = atomSymbols[centerIndex]
                      for direction in range(3):
                          try:
                              dgdx_temp[centerSymbol].append(fpDer[(wrtIndex, wrtSymbol,centerIndex, centerSymbol, direction)])
                          except:
                              pass
                      #TODO: there may exist an atom with 0 neighbor.
                      if len(dgdx_temp[centerSymbol]) > 1:
                          dgdx[centerSymbol].append(np.array(dgdx_temp[centerSymbol]).T.tolist())
                          dEdg_AtomIndices[centerSymbol].append(allfpIndices[keylist[j]][centerIndex])
                          force_AtomIndices[centerSymbol].append([allAtomIndices[keylist[j]][wrtIndex]]*3)
          forceIndices = None
          for element in elements:
              dEdg_AtomIndices[element] = torch.tensor(dEdg_AtomIndices[element]).\
                                           resize_(len(dEdg_AtomIndices[element]), elementFPs[element], 1).\
                                           repeat(1,1,3)
              fp_imageIndices[element] = torch.tensor(fp_imageIndices[element])
              if forceIndices is None:
                  forceIndices = torch.tensor(force_AtomIndices[element])
              else:
                  forceIndices = torch.cat([forceIndices, torch.tensor(force_AtomIndices[element])])
          force_AtomIndices = forceIndices
      return (allElement_fps, fp_imageIndices, fpRange, dgdx, dEdg_AtomIndices, force_AtomIndices,
              natomsPerElement, torch.tensor(natomsPerImage), torch.tensor(natoms), ntotalAtoms)

