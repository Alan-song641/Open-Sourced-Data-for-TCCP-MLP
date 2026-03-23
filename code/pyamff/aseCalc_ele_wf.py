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
from pyamff import zbl

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

class aseCalc_ele_wf(Calculator):
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
                 const=0.0,
                 disp=0.0,
                 exceed_n=0.0, 
                 desire_wf=4, 
                 max_iterations_pot=100, 
                 alpha=0.5, 
                 adjust=0.8, 
                 tolerance=0.1,
                 method='newton',
                 zbl_cut=0.0,
                 zbl_skin=0.0,
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.dir_path = dir_path
        self.modelType = modelType
        
        self.const = const # wf = dE/dn + const
        self.disp = disp # value of dn in dE/dn, if disp=0.0, use analytical wf
        self.exceed_n = exceed_n # initial guess of the excess_n
        self.desire_wf = desire_wf # desire WF
        self.max_iterations_pot = max_iterations_pot 
        self.alpha = alpha # step size for updating excess_n in target potential solving
        self.adjust = adjust # if loss getting higher, lower the gradient by adjust
        self.tolerance = tolerance # convergence criterion of the desire WF
        self.method = method # method to solve target potential, 'secant' or 'newton'
        self.zbl_cut = zbl_cut # cutoff for ZBL potential, unit: Angstrom
        

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
            except Exception as e:
                self.chg_grad = False
            
            self.total_ion_charge = total_ion_charge

            self.model, self.preprocessParas = loadModel_ele(model, fptimer_dict, modelType)

        else:
            self.model = model
            self.preprocessParas = preprocessParas
            # print(preprocessParas)
        
        if self.zbl_cut != 0.0:
            self.zbl_calc = zbl.ZBLCalculator(cutoff=self.zbl_cut, skin=zbl_skin)

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
        # Use a copy to avoid modifying the original atoms object (which causes cache invalidation)
        totnatoms = len(atoms.numbers)
        atoms_work = atoms.copy()
        energy, forces, pred_charges, pred_wf, steps = self.calculateFingerprints(atoms_work)
        
        for ele in pred_charges.keys():
            pred_charges[ele] = pred_charges[ele].data.numpy().flatten()

        self.implemented_properties.append('charges')

        energy = (energy).data.numpy()[0]
        forces = forces.data.numpy()

        if self.zbl_cut != 0.0:
            zbl_energy, zbl_forces, zbl_weights = self.zbl_calc.calculate(atoms)
            energy += zbl_energy
            forces += zbl_forces

        self.results['energy'] = energy
        self.results['forces'] = forces
        
        self.results['pred_wf'] = pred_wf
        self.results['steps'] = steps
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

        self.results['charges'] = char_arr
    
    def get_fps(self, atoms, if_force=True):
        '''Alan: fingerprint filename should be specified'''
        # st1 = time.time()
        self.fpcalc = Fingerprints(uniq_elements=self.Gs.keys(),
                                   filename=os.path.join(self.dir_path, self.fp_name),
                                   nfps=self.nfps)
        
        images = {0: atoms}
        keylist = [0]
        chemsymbols = atoms.get_chemical_symbols()
        '''------------------------------------------'''
        fps, dfps = self.fpcalc.calcFPs(atoms, chemsymbols, if_force)
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

    def calculateFingerprints(self, atoms):
        '''
        Best efficient: newton + analytical wf
        numerical wf is not optimized yet, because it requires two forward passes of the model
        '''
        acf = self.get_fps(atoms, if_force=False) # no need forces here
        efp = Ewald_FPs(structure=atoms, chg_offset=0.0, if_force=False, chg_grad=self.chg_grad)

        if self.method == 'newton':
            pred_wf, pred_N, converged, time, steps = self.target_pot_solve_newton(atoms, acf, efp)
            print(f'[Newton method] time: {time:.2f} seconds, steps: {steps}, pred_wf={pred_wf:.3f}, pred_N={pred_N:.3f}, converged={converged}')
        elif self.method == 'secant':
            pred_wf, pred_N, converged, time, steps = self.target_pot_solve_secant(atoms, acf=None, efp=None)
            print(f'[Secant method] time: {time:.2f} seconds, steps: {steps}, pred_wf={pred_wf:.3f}, pred_N={pred_N:.3f}, converged={converged}')

        if not converged:
            print(f'[Error]: desire_wf={self.desire_wf:.3f} pred_wf={pred_wf:.3f} pred_N={pred_N:.3f}')
            # print(f'[Warning]: Using clipped/initial excess_n={pred_N:.3f} to continue calculation.')
        else:
            # print(f'[Converged]: desire_wf={self.desire_wf:.3f} pred_wf={pred_wf:.3f} pred_N={pred_N:.3f}')
            pass

        atoms = self.set_atom_chg(atoms, pred_N)

        acf = self.get_fps(atoms)

        efp = Ewald_FPs(structure=atoms, real_space_cut=self.real_space_cut, chg_grad=self.chg_grad)

        predEnergies, predForces, pred_charges, charge_percent = self.model(acf.allElement_fps, acf.dgdx, acf, [[efp]])

        return predEnergies, predForces, pred_charges, pred_wf, steps

    def set_atom_chg(self, atoms, excess_n):
        chg = np.zeros(len(atoms)) 
        chg[0] = -excess_n # excess_n = -Q
        atoms.set_initial_charges(chg)

        return atoms

    def _get_omega(self):
        return self.results['energy'] - self.desire_wf * self.exceed_n

    def _get_wf(self, atoms, excess_n, acf=None, efp=None):

        if self.disp == 0.0:
            if acf is None or efp is None:
                acf = self.get_fps(atoms, if_force=False) # no need forces here
                efp = Ewald_FPs(structure=atoms, chg_offset=0.0, if_force=False, chg_grad=self.chg_grad)
            return self._get_wf_analytical(atoms, excess_n, acf, efp)
        else:
            return self._get_wf_numerical(atoms, excess_n)
    
    def _get_wf_numerical(self, atoms, excess_n):

        atoms = self.set_atom_chg(atoms, excess_n)

        acf = self.get_fps(atoms, if_force=False) # no need forces here

        # NOTE: deprecated: because it will change the self.const
        # if excess_n >= 0:
        #     self.disp = abs(self.disp)
        # else:
        #     self.disp = -abs(self.disp)

        # NOTE: mimic partial E / partial n by finite difference, not necessary to be En - En-1
        efp = Ewald_FPs(structure=atoms, chg_offset=0.0, if_force=False, chg_grad=self.chg_grad) # NOTE: Q = sum(atoms.get_int_chg()) + chg_offset
        efp2 = Ewald_FPs(structure=atoms, chg_offset=self.disp, if_force=False, chg_grad=self.chg_grad) 
        predEnergies, predForces, pred_charges, charge_percent = self.model(acf.allElement_fps, acf.dgdx, acf, [[efp]],if_force=False)
        predEnergies2, predForces2, pred_charges2, charge_percent2 = self.model(acf.allElement_fps, acf.dgdx, acf, [[efp2]],if_force=False)
        
        wf = -(predEnergies.data.numpy()[0] - predEnergies2.data.numpy()[0])/self.disp

        return wf + self.const

    def _get_wf_analytical(self, atoms, excess_n, acf=None, efp=None):
        # atoms = self.set_atom_chg(atoms, excess_n)

        # acf = self.get_fps(atoms, if_force=False) # no need forces here 

        # efp = Ewald_FPs(structure=atoms, chg_offset=0.0, if_force=False, chg_grad=False)
        
        efp = copy.deepcopy(efp)
        efp.total_charge = torch.tensor([-excess_n], requires_grad=True) 

        predEnergies, predForces, pred_charges, charge_percent = self.model(acf.allElement_fps, acf.dgdx, acf, [[efp]],if_force=False)
        
        lossgrads = torch.autograd.grad(predEnergies, efp.total_charge, retain_graph=True,
                                            create_graph=True,
                                            allow_unused=True)[0]
        wf = lossgrads.item()

        return wf + self.const
    
    def _get_wf_derivative_analytical(self, atoms, excess_n, acf, efp):
        # atoms = self.set_atom_chg(atoms, excess_n)

        # acf = self.get_fps(atoms, if_force=False) # no need forces here 

        # efp = Ewald_FPs(structure=atoms, chg_offset=0.0, if_force=False, chg_grad=self.chg_grad) 
        
        efp = copy.deepcopy(efp)
        efp.total_charge = torch.tensor([-excess_n], requires_grad=True)

        predEnergies, predForces, pred_charges, charge_percent = self.model(acf.allElement_fps, acf.dgdx, acf, [[efp]],if_force=False)
        
        lossgrads = torch.autograd.grad(predEnergies, efp.total_charge, retain_graph=True,
                                            create_graph=True,
                                            allow_unused=True)[0]
        wf = lossgrads.item()

        # Second derivative
        hessian = torch.autograd.grad(lossgrads, efp.total_charge, retain_graph=True,
                                            create_graph=True,
                                            allow_unused=True)[0]
        slope = -hessian.item()

        return wf + self.const, slope

    def target_pot_solve_newton(self, atoms, acf=None, efp=None):
        '''
        minimize L = (mu(n) - mu_desire)^2 using Newton's method
        '''
        n = copy.deepcopy(self.exceed_n)
        alpha = self.alpha
        previous_loss = None
        
        # Check initial point
        # wf = self._get_wf(atoms, n, acf, efp)

        # if abs(wf - self.desire_wf) <= self.tolerance or self.max_iterations_pot == 0:
        #     return wf, n, True, 0, 0
        
        st = time.time()
        for i in range(self.max_iterations_pot):
            
            wf, slope = self._get_wf_derivative_analytical(atoms, n, acf, efp)
            loss = (wf - self.desire_wf) ** 2
            
            if previous_loss is not None and loss > previous_loss:
                alpha *= self.adjust

            previous_loss = loss
            
            if abs(wf - self.desire_wf) <= self.tolerance:
                return wf, n, True, time.time() - st, i+1
            
            if abs(slope) < 1e-10:
                # print(f"Warning: Vanishing slope {slope} in Newton method")
                return wf, n, False, time.time() - st, i+1

            # Newton update: n_new = n - alpha * f(n)/f'(n)
            delta_n = (wf - self.desire_wf) / slope
            n -= alpha * delta_n

        return wf, n, False, time.time() - st, i+1

    def target_pot_solve_secant(self, atoms, acf=None, efp=None):
        '''
        minimize L = (mu(n) - mu_desire)^2
        '''
        previous_loss = None
        # f_out = open("potential.txt", "a")
        n = copy.deepcopy(self.exceed_n)
        n_all = [n, n + 0.1]
        workfunction = [0, 0]

        workfunction1 = self._get_wf(atoms, excess_n=n_all[0], acf=acf, efp=efp)
        
        workfunction[0] = workfunction1
        # line = f"\n Step    N       mu         \n "
        # f_out.write(line.ljust(10))
        # line = f"step 0: {round(n_all[0], 3)} {round(workfunction1, 3)}"
        # f_out.write(line.ljust(10) + "\n")
        # print(np.max(forces1))
        if abs(workfunction[0] - self.desire_wf) <= self.tolerance or self.max_iterations_pot == 0:
            return workfunction1, n_all[0], True, 0, 0
        
        st = time.time()
        alpha = self.alpha
        for i in range(self.max_iterations_pot):
            workfunction2 = self._get_wf(atoms, excess_n=n_all[1], acf=acf, efp=efp)
            
            # print(np.max(forces2))
            workfunction[1] = workfunction2
            loss = (workfunction2 - self.desire_wf) ** 2

            # line = f"step {i}: {round(n_all[1], 3)} {round(workfunction2, 3)}"
            #print(n_all[1], n_all[0])
            # f_out.write(line.ljust(10) + "\n")

            if abs(workfunction[1] - self.desire_wf) <= self.tolerance:
                return workfunction2, n_all[1], True, time.time() - st, i+1
            
            # delta \phi / delta n
            grad_mu_n = (workfunction[1] - workfunction[0]) / (n_all[1] - n_all[0])

            if abs(grad_mu_n) < 1e-10:
                return workfunction2, n_all[1], False, time.time() - st, i+1

            delta_n = (workfunction[1] - self.desire_wf) / grad_mu_n
            n_all[0] = n_all[1]
            n_all[1] -= alpha * delta_n

            if previous_loss is not None and loss > previous_loss:
                alpha *= self.adjust

            previous_loss = loss
            workfunction[0] = workfunction[1]

        return workfunction2, n_all[1], False, time.time() - st, i+1

# chg-ele
if __name__ == "__main__":
    # profiler = profile_functions()
# 
    from ase.io import read, write
    # import cProfile 
    dir_path = '/mnt/d/Alan/PhD/pyamff_training/QEQ/Cucc/qeq-v2/nonGC/train6283/train04'
    model_path = dir_path + "/pyamff.pt"
    atoms_ele = read(dir_path + '/wf3.2disp0.01/neb_images_nonGC.traj', index=":")
    for i in range(len(atoms_ele)):
        atoms_ele[i].calc = aseCalc_ele_wf(dir_path, model_path, desire_wf=3.2)
        atoms_ele[i].get_forces()
    
    write(dir_path + '/wf3.2disp0.01/neb_images_nonGC_with_chg.extxyz', atoms_ele)

    # def test():
    #     for i in range(20):
    #         atoms_ele[i].calc = aseCalc_ele_wf(dir_path, model_path)
    #         atoms_ele[i].get_forces()
    
    # cProfile.runctx('test()',globals(),locals(),filename='profile.bin')


