"""
This calculator is used to load a trained machine-learning model and do calculations
"""
from __future__ import division

import numpy as np
import os

from ase.calculators.calculator import Calculator, all_changes
from pyamff.fingerprints.ewald_FPs import Ewald_FPs
from pyamff.mlModels.pytorchNN import NeuralNetwork
# from pyamff.mlModels.pytorchNN import NeuralNetworkFeatureImportanceEle
from pyamff.utilities.preprocessor import normalizeParas
from pyamff.fingerprints.fingerprints import Fingerprints
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
import torch, sys, time
from pyamff.config import ConfigClass
from pyamff import fmodules
from pyamff.config import G1, G2
from configparser import ConfigParser
import copy

def loadModel_ele(model_path, fp_timer, modelType='NeuralNetwork'):

    # Alan: This three lines is no used
    # from pyamff.utilities.preprocessor import Scaler
    # scaler = Scaler(scalerType='STDScaler')
    # scaler = scaler.set_scaler()
    loaded = torch.load(model_path, weights_only=False)
    modelParameters = loaded['Modelparameters']

    output_layer_bias_key = next(key for key in loaded['state_dict'].keys() if key.endswith('.outputLayer.bias'))
    bias_length = len(loaded['state_dict'][output_layer_bias_key])
    if bias_length == 3:
        ifhardness = True
    else:
        ifhardness = False

    # print(model_path)
    model = NeuralNetwork(
        hiddenlayers=modelParameters['hiddenlayers'],
        activation=modelParameters['activation'],
        nFPs=modelParameters['nFPs'],
        forceTraining=modelParameters['forceTraining'],
        #slope=modelParameters['slope'],
        # Alan: Scaler info should be read from torch.load(model_path)
        scaler=modelParameters['scaler'],
        # scaler=scaler,
        ifhardness=ifhardness,
        # input_charge_epoch=loaded['Electronegativity'],
        fp_timer = fp_timer,
        )
    model.load_state_dict(loaded['state_dict'])
    # print(loaded['preprocessParas'])
    return model, loaded['preprocessParas']

class aseCalc_ele(Calculator):
    '''
    This class is for pyamff-ewald
    '''
    implemented_properties = ['energy', 'forces']
    default_parameters = {}
    nolabel = True

    def __init__(self,
                 dir_path,
                 model,
                 fp_name='fpParas.dat',
                 modelType='NeuralNetwork',
                 preprocessParas=None,
                 config_name='config.ini',
                 total_ion_charge=None,
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.dir_path = dir_path
        self.modelType = modelType
        if preprocessParas is None:
            self.config = ConfigParser()
            self.config.read(self.dir_path+'/'+config_name)

            try:
                self.real_space_cut = float(self.config['Main']['real_space_cut'])
            except:
                self.real_space_cut = None
            
            try:
                fptimer = self.config['Main']['fp_timer']
            except:
                fptimer = 'none'
            
            try:
                fptimer_a = float(self.config['Main']['fp_timer_a'])
            except:
                fptimer_a = 1.0
                
            fptimer_dict = {'fptimer':fptimer, 'alpha':fptimer_a}

            try:
                self.chg_grad = bool(self.config['Main']['chg_grad'])
            except:
                self.chg_grad = False
            
            self.total_ion_charge = total_ion_charge

            self.model, self.preprocessParas = loadModel_ele(model, fptimer_dict, modelType)

        else:
            self.model = model
            self.preprocessParas = preprocessParas
            # print(preprocessParas)
        self.Gs = self.preprocessParas['fingerprints'].fp_paras
        # print('Gs:', self.Gs)
        self.nfps = {}
        for key in self.Gs.keys():
            self.nfps[key] = len(self.Gs[key])
        #self.Gs = self.preprocessParas['fingerprints']
        self.fprange = self.preprocessParas['fpRange']
        self.intercept = self.preprocessParas['intercept']
        # print('keys', self.Gs.keys())
        self.fp_name = fp_name
        self.ttime = 0

    def calculate(self,
                  atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):


        Calculator.calculate(self, atoms, properties, system_changes)
        totnatoms = len(atoms.numbers)
        atoms_work = atoms.copy()
        energy, forces, pred_charges, charge_percent = self.calculateFingerprints(atoms_work)
        
        for ele in pred_charges.keys():
            pred_charges[ele] = pred_charges[ele].data.numpy().flatten()
        for ele in charge_percent.keys():
            charge_percent[ele] = charge_percent[ele].data.numpy().flatten()

        self.implemented_properties.append('charges')

        if self.model.adjust and self.model.scalerType in [
                'LinearScaler', 'MinMaxScaler'
        ]:
            # Jiyoung:should not add intercept because it is already added in pytorchNN.py
            self.results['energy'] = (energy).data.numpy()[0]
            self.results['forces'] = forces.data.numpy()

        elif self.model.scalerType in ['STDScaler']:
            self.results['energy'] = (
                self.model.slope * energy +
                self.intercept * totnatoms).data.numpy()[0]
            #TODO: For STDScaler, sigma_F should be multiplied
            self.results['forces'] = (self.model.f_std * forces).data.numpy()

        elif self.model.scalerType in ['NoScaler']:
            self.results['energy'] = (energy).data.numpy()[0]
            self.results['forces'] = forces.data.numpy()
        else:
            self.results['energy'] = (energy + self.intercept).data.numpy()[0]
            self.results['forces'] = forces.data.numpy()
        
        # self.results['charges'] = pred_charges
        # self.results['charge_percent'] = charge_percent
        self.set_charge(pred_charges)

    def set_charge(self, pred_charges):
        # pred_charges = self.results['charges']
        # charge_percent = self.results['charge_percent']

        chem_symbol = self.atoms.get_chemical_symbols()
        chem_set = list(set(chem_symbol))
        chem_set.sort(key=chem_symbol.index)

        char_arr = [pred_charges[element] for element in chem_set]
        char_arr = np.concatenate(char_arr)
        
        # self.atoms.set_initial_charges(char_arr)
        # print('Charges set to:', char_arr)

        self.results['charges'] = char_arr
    
    def get_fps(self, atoms=None):
        '''Alan: fingerprint filename should be specified'''
        # st1 = time.time()
        self.fpcalc = Fingerprints(uniq_elements=self.Gs.keys(),
                                   filename=os.path.join(self.dir_path, self.fp_name),
                                   nfps=self.nfps)
        
        images = {0: atoms}
        keylist = [0]
        chemsymbols = atoms.get_chemical_symbols()
        '''------------------------------------------'''
        fps, dfps = self.fpcalc.calcFPs(atoms, chemsymbols)
        fmodules.fpcalc.cleanup()

        acf = atomCenteredFPs()
        acf.sortFPs(chemsymbols,
                    fps,
                    self.nfps,
                    properties=None,
                    keylist=keylist,
                    fingerprintDerDB=dfps)
        
        acf.stackFPs_calc(acf)
        
        fprange, magnitudeScale, interceptScale = normalizeParas(self.fprange)

        acf.normalizeFPs(fprange, magnitudeScale, interceptScale)
        # st2 = time.time()
        # print('FP TIME:', st2 - st1)
        return acf

    def get_efp(self, atoms=None):
        efp = Ewald_FPs(structure=atoms, real_space_cut=self.real_space_cut, total_ion_charge=self.total_ion_charge, chg_grad=self.chg_grad)
        return efp

    def calculateFingerprints(self, atoms=None):
        acf = self.get_fps(atoms)

        efp = self.get_efp(atoms)

        # st3 = time.time()
        predEnergies, predForces, pred_charges, charge_percent = self.model(acf.allElement_fps, acf.dgdx, acf, [[efp]])

        # print('ML TIME:', time.time() - st3)

        return predEnergies, predForces, pred_charges, charge_percent
    




# chg-ele
if __name__ == "__main__":
    # profiler = profile_functions()
# 
    from ase.io import read
    import cProfile 
    dir_path = '/home/alan/Github/pyamff-cv/examples/chg-PdO'
    model_path = "/home/alan/Github/pyamff-cv/examples/chg-PdO/pyamff_100.pt"
    atoms_ele = read('/home/alan/Github/pyamff-cv/examples/chg-PdO/test.traj', index=":")
    def test():
        for i in range(20):
            atoms_ele[i].calc = aseCalc_ele(dir_path, model_path)
            atoms_ele[i].get_forces()
    
    cProfile.runctx('test()',globals(),locals(),filename='profile.bin')


