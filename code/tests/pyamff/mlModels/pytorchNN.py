#from . import generateTensorFlowArrays
#from .. import FileDatabase
from collections import OrderedDict
from re import T
import numpy as np

import torch
from torch.utils import data
from scipy.stats import truncnorm
import torch.nn as nn
from torch.nn import Linear
from pyamff.mlModels.lossFunctions import LossFunction
from pyamff.utilities.truncatedNormal import TruncatedNormal
from pyamff.utilities.fileIO import saveData
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs

# https://ptable.com/?lang=zh-hans
en_dict = {'H': 2.20,'He': 4.16,'Li': 0.98,'Be': 1.57,'B': 2.04,'C': 2.55,
            'N': 3.04,'O': 3.44,'F': 3.98,'Ne': 4.79,'Na': 0.93,'Mg': 1.31,
            'Al': 1.61,'Si': 1.98,'P': 2.19,'S': 2.58,'Cl': 3.16,'Ar': 3.24,
            'K': 1.12,'Ca': 1.01,'Sc': 1.36,'Ti': 1.54,'V': 1.63,'Cr': 1.66,
            'Mn': 1.55,'Fe': 1.83,'Co': 1.88,'Ni': 1.92,'Cu': 1.90,'Zn': 1.65,
            'Ga': 1.81,'Ge': 2.01,'As': 2.18,'Se': 2.55,'Br': 2.96,'Kr': 3.00,
            'Rb': 0.82,'Sr': 0.95,'Y': 1.22,'Zr': 1.33,'Nb': 1.59,'Mo': 2.16,
            'Tc': 1.91,'Ru': 2.20,'Rh': 2.28,'Pd': 2.20,'Ag': 1.93,'Cd': 1.69,
            'In': 1.78,'Sn': 1.96,'Sb': 2.05,'Te': 2.12,'I': 2.66,'Xe': 2.60,
            'Cs': 0.79,'Ba': 0.89,'Hf': 1.32,'Ta': 1.51,'W': 2.36,'Re': 1.93,
            'Os': 2.18,'Ir': 2.20,'Pt': 2.28,'Au': 2.54,'Hg': 2.00,'Ti': 1.62,
            'Pb': 2.33,'Bi': 2.02,'Po': 1.99,'At': 2.22,'Rn': 2.43
        }

# radius_dict = {
#             #    'O': 0.48,
#             #    'Ti': 1.76,
#                'O': 0.60,
#                'Ti': 1.40,

#                'H': 0.25,
#                'C': 0.70,
#                'N': 0.65,
#                'F': 0.50,
#                'S': 1.00,
#                'V': 1.35,
#                'W': 1.35,
#                'Ta': 1.45,
#                'Si': 1.10,
#                'Ni': 1.35,
#                'Au': 1.35,
#                'Ag': 1.60,
#                'Ge': 1.25,
#                'Pd': 1.40,
#                'Pt': 1.35,
#                'Cu': 1.45,
#                 }

# '''
# NOTE: NOW I want the J(atomic hardness) also the function of g
# '''

# # J0 = 14.4/R0 --> Charge equilibration for molecular dynamics simulations. J. Phys. Chem. 95, 3358–3363 (1991)
# hardness_dict = dict()

# # CONV_FACT = 1e10 * constants.e / (4 * pi * constants.epsilon_0) # 14.4
# CONV_FACT = 1
# for key in radius_dict.keys():
#     hardness_dict[key] = CONV_FACT / (radius_dict[key])


# # # 1. Putz, M. V. International Journal of Quantum Chemistry 106, 361–389 (2006). Table V
# # # 2. Kumari, V. et al. J Math Chem 60, 360–372 (2022).
# en_dict = {
#     "H": 7.18, "He": 12.27, "Li": 3.02, "Be": 3.43, "B": 4.26, "C": 6.24, "N": 6.97, "O": 7.59, "F": 10.4, "Ne": 10.71,
#     "Na": 2.80, "Mg": 2.6, "Al": 3.22, "Si": 4.68, "P": 5.62, "S": 6.24, "Cl": 8.32, "Ar": 7.7,
#     "K": 2.39, "Ca": 2.29, "Sc": 3.43, "Ti": 3.64, "V": 3.85, "Cr": 3.74, "Mn": 3.85, "Fe": 4.26, "Co": 4.37, "Ni": 4.37, "Cu": 4.47, "Zn": 4.26,
#     "Ga": 3.22, "Ge": 4.58, "As": 5.3, "Se": 5.93, "Br": 7.59, "Kr": 6.86,
#     "Rb": 2.29, "Sr": 1.98, "Y": 3.43, "Zr": 3.85, "Nb": 4.06, "Mo": 4.06, "Tc": 3.64, "Ru": 4.06, "Rh": 4.26, "Pd": 4.78, "Ag": 4.47, "Cd": 4.16,
#     "In": 3.12, "Sn": 4.26, "Sb": 4.89, "Te": 5.51, "I": 6.76, "Xe": 5.82
# }

# # # Finite difference hardness: ηFD
hardness_dict = {
    "H": 6.45, "He": 12.48, "Li": 4.39, "Be": 5.93, "B": 4.06, "C": 4.99, "N": 7.59, "O": 6.14, "F": 7.07, "Ne": 10.92,
    "Na": 2.89, "Mg": 4.99, "Al": 2.81, "Si": 3.43, "P": 4.89, "S": 4.16, "Cl": 4.68, "Ar": 8.11,
    "K": 1.12, "Ca": 3.85, "Sc": 3.22, "Ti": 3.22, "V": 2.91, "Cr": 3.12, "Mn": 3.64, "Fe": 3.64, "Co": 3.43, "Ni": 3.22, "Cu": 3.22, "Zn": 5.2,
    "Ga": 2.81, "Ge": 3.33, "As": 4.47, "Se": 3.85, "Br": 4.26, "Kr": 7.28,
    "Rb": 1.87, "Sr": 3.74, "Y": 2.91, "Zr": 3.02, "Nb": 2.91, "Mo": 3.12, "Tc": 3.64, "Ru": 3.43, "Rh": 3.22, "Pd": 3.64, "Ag": 3.12, "Cd": 4.78,
    "In": 2.70, "Sn": 3.02, "Sb": 3.85, "Te": 3.54, "I": 3.74, "Xe": 6.34, "Pb": 2.88, "Au": 3.02,
}

for e in hardness_dict:
    hardness_dict[e] = hardness_dict[e] / 2.5

# in the unit of ev/atom
cohe_dict = { "H": 0.34, "He": 1.0, "Li": 1.63, "Be": 3.32, "B": 5.81, "C": 7.37, "N": 4.92, "O": 2.6, "F": 0.84, "Ne": 0.02, 
             "Na": 1.113, "Mg": 1.51, "Al": 3.39, "Si": 4.63, "P": 3.43, "S": 2.85, "Cl": 1.4, "Ar": 0.08,
            "K": 0.934, "Ca": 1.84, "Sc": 3.9, "Ti": 4.85, "V": 5.31, "Cr": 4.1, "Mn": 2.92, "Fe": 4.28, "Co": 4.39, "Ni": 4.44, "Cu": 3.49, "Zn": 1.35,
             "Ga": 2.81, "Ge": 3.85, "As": 2.96, "Se": 2.46, "Br": 1.22, "Kr": 0.116, 
             "Rb": 0.852, "Sr": 1.72, "Y": 4.37, "Zr": 6.25, "Nb": 7.57, "Mo": 6.82, "Tc": 6.85, "Ru": 6.74, "Rh": 5.75, "Pd": 3.89, "Ag": 2.95, "Cd": 1.16,
               "In": 2.52, "Sn": 3.14, "Sb": 2.75, "Te": 2.19, "I": 1.11, "Xe": 0.16, "Pb": 2.03, "Au": 3.81,
}

def batchGenerator(acfs):
    if len(acfs) == 1:
        batch = atomCenteredFPs()
        batch.stackFPs(acfs)
        return batch
    batch = atomCenteredFPs()
    batch.stackFPs(acfs)

    return batch

def batchGenerator_ele(efps):
    '''
    efps:OrderDict{0:efp, 1:efp ...}
    '''
    return efps

class LearnableScalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.val = nn.Parameter(torch.tensor(init_value).double())

    def forward(self):
        return self.val

class NeuralNetwork(nn.Module):
    """
    hiddenlayers: define the structure of the neural network.
                 i.e. (2,3) define a neural network with two hidden layers,
                            containing 2 and 3 neurons, respectively
    nFPs: a dictionary that defines number of fingerprints for each element.
          i.e. {'H': 3, 'Pd':2}: 3 and 2 fingerprints for 'H' and 'Pd', respectively
    maxEpochs: maximum epochs the training process will take
    energyCoefficient: the weight of energy in the loss function
    forceCoefficient: the weight of force in the loss function
    params: a list or array of parameters will be used to initilize the NN model
          i.e., [ #params for 'H': fp1: fingerprint 1, n1: neuron 1 in the corresponding layer
                  [[fp1_n1, fp2_n1, fp3_n1], [fp1_n2, fp2_n2, fp3_n2]],   #weights connecting inputlayer to hiddenlayer 1
                  [ bias_n1,                  bias_n2],                   #bias on hiddenlayer 1
                  [[n1_n1, n2_n1], [n1_n2, n2_n2], [n1_n3, n2_n3]],       #weights connecting hiddenlayer 1 to hiddenlayer 2
    e             [ bias_n1,       bias_n2,         bias_n3],             #bias on hiddenlayer 2
                  [[n1_n1, n2_n1, n3_n1]],                                #weights connecting hiddenlayer 1 to outputlayer
                  [ bias_n1],                                             #bias on outputlayer 

                  #params for 'Pd': fp1: fingerprint 1, n1: neuron 1 in the corresponding layer
                  [[fp1_n1, fp2_n1], [fp1_n2, fp2_n2]],                   #weights connecting inputlayer to hiddenlayer 1
                  [ bias_n1,          bias_n2],                           #bias on hiddenlayer 1
                  [[n1_n1, n2_n1], [n1_n2, n2_n2], [n1_n3, n2_n3]],       #weights connecting hiddenlayer 1 to hiddenlayer 2
                  [ bias_n1,       bias_n2,         bias_n3],             #bias on hiddenlayer 2
                  [[n1_n1, n2_n1, n3_n1]],                                #weights connecting hiddenlayer 1 to outputlayer
                  [ bias_n1],                                             #bias on outputlayer 
                ]
   """
    def __init__(self,
                hiddenlayers = (5, 5),
                nFPs = {'Au': 10, 'H':5},
                activation='sigmoid',
                forceTraining = True,
                cohE = 0.0, # Es part, [0.0, 1.0]
                en_frac = 1.0, # electronegativity part [0.0001, 1.0]
                params = None,
                scaler = None,
                debug = False,
                initial_weights = 'random_truncated_normal',
                # for ifElectronegativity
                partitions=None,
                ifElectronegativity=True,
                ifhardness=True,
                if_short=True,
                if_long=True,
                process_number=None,
                fp_timer='none',
                #slope = None,
                #energyRange = None,
                #forceRange = None
                ):
        super().__init__()
        self.hiddenlayers = hiddenlayers
        self.activation = activation
        self.n_layers = len(hiddenlayers) + 2
        if activation == 'sigmoid':
           self.actF = nn.Sigmoid()
        if activation == 'relu':
           self.actF = nn.ReLU()
        if activation == 'tanh':
           self.actF = nn.Tanh()
        if activation == 'softplus':
           self.actF = nn.Softplus()
        # if activation == 'silu': # perform not well after testing
        #    self.actF = nn.SiLU()
        if activation == 'reluTanh': # ???
           self.actF_1 = nn.ReLU()
           self.actF_2 = nn.Tanh()
        self.nFPs = nFPs
        self.elements = np.array([element for element in nFPs.keys()])
        self.nn_models = {}
        self.hd_names = {}
        self.model_params = []
        self.model_namedparams = []
        self.debug = debug
        self.initial_weights = initial_weights

        self.if_short = if_short
        self.if_long = if_long
        self.ifhardness = ifhardness
        '''
        -- If fp_timer is 'one', then alpha is used to determine how many dimensions will be concatenated to the original fps. For example, if alpha is 10, then 10 dimensions will be concatenated to the original fps, and the input layer will have n_Gs+10 neurons. 
        -- If fp_timer is 'exp', then alpha is used to determine the exponential decay of the timer, and only 1 dimension will be concatenated to the original fps, and the input layer will have n_Gs+1 neurons.
        '''
        if fp_timer != 'none':
            self.fp_timer = fp_timer['fptimer']
            self.fp_timer_alpha = fp_timer['alpha']
            if self.fp_timer == 'one':
                self.fp_timer_alpha = int(self.fp_timer_alpha)
        else:
            self.fp_timer = 'none'
            self.fp_timer_alpha = 1.0

        #print('  paras',list(self.nn_models.parameters()))
        self.forceTraining = forceTraining
        # Alan: changed this tag for Es_frac, for initialization of NN bias
        self.Es_frac = cohE
        self.en_frac = en_frac
        self.scaler = scaler
        self.scalerType = scaler.scalerType
        self.adjust = False
        if self.scalerType == 'NoScaler':
            self.slope = torch.tensor(scaler.slope)
            self.intercept = torch.tensor(scaler.intercept)
        if self.scalerType in ['LinearScaler', 'MinMaxScaler']:
            self.adjust = scaler.adjust
            if self.adjust:
                self.slope = nn.parameter.Parameter(torch.tensor(scaler.slope))
                self.intercept = nn.parameter.Parameter(torch.tensor(scaler.intercept))
            else:
                self.slope = torch.tensor(scaler.slope)
                self.intercept = torch.tensor(scaler.intercept)
            self.fc_slope = scaler.fc_slope
            self.fc_intercept = scaler.fc_intercept
        if self.scalerType in ['LogScaler']:
            self.eMinMax = scaler.eMinMax
            self.eRange = self.eMinMax[0] - self.eMinMax[1]
            #self.fMinMax = scaler.fMinMax
        if self.scalerType in ['STDScaler']:
            self.intercept = torch.tensor(scaler.intercept)
            self.slope = torch.tensor(scaler.slope)
            self.e_std = scaler.slope
            self.f_std = scaler.f_std
            self.ef_coef = self.e_std/self.f_std

        self.imageIndices = None
        self.nimages = None
        self.fp_d = None
        self.dEdg_AtomIndices = None
        self.force_AtomIndices = None
        self.natomsPerElement = None
        self.ntotalAtoms = None

        self.scalerType = scaler.scalerType
        self.partitions = partitions
        self.params = params
        self.process_number = process_number

        # NOTE: means this is for alan_csv function!!
        if self.params == 'restart' or self.partitions is None: 
            self.params = None
            # NOTE: original code flow is adopted for alan_csv.py
            for element, n_Gs in self.nFPs.items():
                self.hd_names[element] = []
                self.atomModel(n_Gs, element)
                #self.model_params += list(self.nn_models[element].parameters())
                #self.model_namedparams += list(self.nn_models[element].named_parameters())

            self.nn_models = nn.ModuleDict(self.nn_models)
            # self.setParams(self.params)
            for element in self.elements:
                self.nn_models[element].double()
            
        else:
            if ifElectronegativity:
                print('make the predicted X as referred value')
                self.init_reference_value()

            else:
                print('Refer Electro-negativity to repeat initialization of NN')
                self.init_param() # NOTE: repeat initialization until satisfied the requirement


    def setParams(self, params):
        i=0
        # mean, stddev, lowerbound, upbound
        #params_to_dump = []
        if params is not None: #if initital weights and biases given
            for param in self.parameters():
                param.data = torch.tensor(params[i]).double()
                #print('shape', param.data.shape)
                i+=1
        else: # we are picking initial weights
            tn = TruncatedNormal(torch.Tensor([0.0]), torch.Tensor([0.1]), torch.Tensor([-0.5]), torch.Tensor([0.5]))
            #lower, upper = 3.5, 6
            #mu, sigma = 5, 0.7
            #X = stats.truncnorm(
            #    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            #for param in self.parameters(): 
                #param.data = torch.tensor(truncnorm.rvs(-10, 10, loc=0.0, scale=0.1, size=list(param.data.shape)))
            #    param.data = tn.rsample(param.data.shape)
            #    params_to_dump.append(param.data)
            #with open("saved_model_params.pyamff", "wb") as f:
            #    pickle.dump(params_to_dump, f)
            
            #self.initial_weights ='sqrt_prev_layer_num_neurons'
            if self.initial_weights =='sqrt_prev_layer_num_neurons': # scale initial weights by root(num_neurons)
                #print ('scaling initial weights with nn layer size')
                #print ('self.parameters: ',self.parameters)
                #print ('self.parameters: ',)
                num_neurons_in_previous_layer = 1 #for input layer, we randomly sample, no previous layer exists to scale w.r.t
                # would it be better to scale w.r.t number of fingerprints for the input layer or is this better?
                for param in self.parameters(): #for each parameter set
                    #print ('param.data.shape: ',param.data.shape)
                    #print ('param: ',param)
                    if len(param.data.shape) !=1: #if weights (i.e do not want to scale inital biases, only initial weights)
                        #print ('nnipl: ',num_neurons_in_previous_layer)
                        param.data = tn.rsample(param.data.shape)/(np.sqrt(num_neurons_in_previous_layer))
                        #num_neurons_in_previous_layer = param.data.shape[1]
                    else: #else it is a bias initialization
                        num_neurons_in_previous_layer = param.data.shape[0] #get the number of neurons
                        param.data = tn.rsample(param.data.shape)
            elif self.initial_weights =='random_uniform': # pick uniform weights
                #print ("uniform random")
                for param in self.parameters():
                    # by default pytorch produces numbers randomly uniform on [0,1)
                    # so shifting it to produce uniform numbers on [-1, 1)
                    param.data = 2*torch.rand(param.data.shape)-1
                    #print (param.data.shape)
            else:#do not scale initial weights
                #print ('default random weights else')
                for param in self.parameters():
                    #param.data = 0*tn.rsample(param.data.shape)
                    param.data = tn.rsample(param.data.shape)
    
    """
    Alan: output nodes changed into 3 (Ei, Xi, Ji)
    atomModel: define the NN model for each atom.
    It will generate a model dictionary with 'element' 
    as the key and a inner dictionary with the name
    of each layer as the key
    """
    def atomModel(self, n_Gs, element):
        if self.fp_timer != 'one':
            self.nn_models[element] = nn.ModuleDict({'inputLayer':Linear(n_Gs, self.hiddenlayers[0]).double()})
        else:
            # additional one dimension concatenated to fps
            self.nn_models[element] = nn.ModuleDict({'inputLayer':Linear(n_Gs+self.fp_timer_alpha, self.hiddenlayers[0]).double()})
        for i in range(len(self.hiddenlayers)-1):
            self.hd_names[element].append('hiddenLayer_'+str(i+1))
            self.nn_models[element][self.hd_names[element][i]] = \
                   Linear(self.hiddenlayers[i], self.hiddenlayers[i+1]).double()
        
        # Alan: output nodes changed into 3 (Ei, Xi, Ji)
        if self.ifhardness:
            self.nn_models[element]['outputLayer'] = Linear(self.hiddenlayers[-1], 3).double()
        else:
            self.nn_models[element]['outputLayer'] = Linear(self.hiddenlayers[-1], 2).double()

        if 'linear' in self.fp_timer:
            # X_weight as a seperate trainable parameter for each element 
            self.nn_models[element]['X_weight'] = LearnableScalar(1.0)
            self.nn_models[element]['X_bias'] = LearnableScalar(0.0)
            self.nn_models[element]['J_weight'] = LearnableScalar(1.0) # deprecated 
            self.nn_models[element]['J_bias'] = LearnableScalar(0.0) # deprecated

    def set(self, imageIndices,nimages,
        dEdg_AtomIndices,force_AtomIndices,
        natomsPerElement, ntotalAtoms, slope=None):

        self.imageIndices = imageIndices
        self.nimages = nimages
        self.dEdg_AtomIndices = dEdg_AtomIndices
        self.force_AtomIndices = force_AtomIndices
        self.natomsPerElement = natomsPerElement
        self.ntotalAtoms = ntotalAtoms
        if self.slope is None:
            self.slope = slope

    def init_param(self):
        '''
        Alan: repeat the initialzation of NN parameters until the pred_charges 
            can correspond the Electronegativity of the element
            partition=None,
            ifElectronegativity=True,
            process_number=40
        '''

        all_satisfied = False
        find_iter = 0

        while not all_satisfied:

            # NN re-initialization
            for element, n_Gs in self.nFPs.items():
                self.hd_names[element] = []
                self.atomModel(n_Gs, element)
            self.nn_models = nn.ModuleDict(self.nn_models)
            self.setParams(self.params)
            for element in self.elements:
                self.nn_models[element].double()
            
            for rank in range(self.process_number):

                partition = self.partitions.use(rank)
                partition_Ele = self.partitions.use_Ele(rank)

                # generate data loader (batch size = 1)
                batches = torch.utils.data.DataLoader(
                    partition,
                    batch_size=1,
                    collate_fn=batchGenerator,
                    shuffle=False)

                batches_Ele = torch.utils.data.DataLoader(
                    partition_Ele,
                    batch_size=1,
                    collate_fn=batchGenerator_ele,
                    shuffle=False)

                # data loader (batch size = 1)
                for i, data in enumerate(zip(batches, batches_Ele)):
                    # Alan: call remaked forward() function
                    batch = data[0]
                    batch_ele = data[1]

                # call forward of NN
                # charges, Es = self.forward_init(fps=batch.allElement_fps)
                energies, forces, charges, charge_percent = self.forward(
                    batch.allElement_fps,
                    batch.dgdx,
                    batch,
                    batch_ele,
                    'cpu',
                    nIter=0,
                    logger=None)

                Es = charge_percent['Es']
                Ewald_tot = charge_percent['Ewald_tot']
                Epercent = charge_percent['Epercent']

                elements = list(charges.keys())

                for element in elements:
                    # detect float(torch.mean(charges[element])) is a float 
                    if not np.isnan(float(torch.mean(charges[element]))):
                        charges[element] = float(torch.mean(charges[element]))
                    else:
                        # remove this key from charges
                        del charges[element]

                # charges_dict = {}
                # for element in self.nFPs.keys():
                #     charges_dict[element] = float(torch.mean(charges[element]))
                charge_sort = sorted(zip(charges.values(), charges.keys()))

                '''
                Alan: satisfied requirement #1: 
                Qi < 0, Qj > 0 if electronegative i > j
                '''

                en_sort = {}
                for element in charges.keys():
                    en_sort[element] = en_dict[element]
                en_sort = sorted(zip(en_sort.values(), en_sort.keys()), reverse=True) # larger the electronegative value, smaller the charge of the element

                charge_flag = True
                charge_list = []
                # Judge the value relationship of charge_sort == en_sort
                for i in range(len(en_sort)):
                    if en_sort[i][1] != charge_sort[i][1]:
                        charge_flag = False
                    charge_list.append(charge_sort[i][0])
                charge_list = np.array(charge_list)

                # Judge if their abs value is larger than q_threshold
                q_threshold = 0.1
                if np.all(np.abs(charge_list) > q_threshold):
                    pass
                else:
                    charge_flag = False

                # if the batch in this loop do not satisfied EN, break to re-initialization NN
                if charge_flag == False:
                    print('[{}]  Charges Stop at index {}: [info] {}'.format(find_iter + 1, rank, charges))
                    break
                else:
                    print('[{}]  Charges OKAY at index {}: [info] {}'.format(find_iter + 1, rank, charges))

                '''
                Alan: satisfied requirement #2: 
                Es < -0
                '''
                # Es_flag = True
                # if Es < -1e-8:
                #     pass
                # else:
                #     Es_flag = False

                # if Es_flag == False:
                #     print('[{}]  Es Stop at index {}: [info] {}'.format(
                #         find_iter + 1, rank, Es))
                #     break
                '''
                Alan: satisfied requirement #3: 
                0 % < Epercent < 75 %
                '''
                # Ewald_tot_flag = True
                # if 0 < Epercent < 75:
                #     pass
                # else:
                #     Ewald_tot_flag = False

                # if Ewald_tot_flag == False:
                #     print('[{}]  Epercent Stop at index {}: [info] {}'.format(
                #         find_iter + 1, rank, Epercent))
                #     break

            # stop iteration when all_satisfied = True
            # all_satisfied = charge_flag and Es_flag and Ewald_tot_flag
            all_satisfied = charge_flag

            find_iter += 1

        return find_iter, charge_sort, Es, Epercent

    def init_reference_value(self):
        '''
        Alan:initialzation of NN bias of the last layer to make the X as referred value
        '''
        
        # NN re-initialization
        for element, n_Gs in self.nFPs.items():
            self.hd_names[element] = []
            self.atomModel(n_Gs, element)

        self.nn_models = nn.ModuleDict(self.nn_models)

        self.setParams(self.params)
        
        for element in self.elements:
            self.nn_models[element].double()

        tuned_tag = {}
        for e in self.elements:
            tuned_tag[e] = False

        for rank in range(self.process_number):
            partition = self.partitions.use(rank)
            partition_Ele = self.partitions.use_Ele(rank)

            # generate data loader (batch size = 1)
            batches = torch.utils.data.DataLoader(partition, batch_size=1, collate_fn=batchGenerator, shuffle=False)
            batches_Ele = torch.utils.data.DataLoader(partition_Ele, batch_size=1, collate_fn=batchGenerator_ele, shuffle=False)

            # data loader
            for i, data in enumerate(zip(batches, batches_Ele)):
                # Alan: call remaked forward() function
                batch = data[0]
                batch_ele = data[1]

            eles = np.unique([[] + list(b[0].chem_num_dict.keys()) for b in batch_ele]) # all of the elements in this batch

            if all([tuned_tag[e] for e in eles]): # only forward the batch if any element not tuned yet
                continue
            print(rank, eles)

            # possible cohesive energy reference
            ref_eng = 0
            natoms = 0
            for e in self.elements:
                ref_eng += cohe_dict[e] * batch.natomsPerElement[e]
                natoms += batch.natomsPerElement[e]
            # ref_const_eng = (ref_eng - torch.sum(batch.energies))/natoms
            ref_ratio = float(ref_eng / torch.sum(batch.energies).item())

            # call forward of NN
            # charges, Es = self.forward_init(fps=batch.allElement_fps)
            energies, forces, charges, charge_percent = self.forward(batch.allElement_fps, batch.dgdx, batch, batch_ele, 'cpu', logger=None)

            # print(torch.mean(batch.energies))
            for element in eles:
                print('{} info before re-initialize: q: {:.2f}, X: {:.2f}, J: {:.2f}, Es: {:.2f}, Eele: {:.2f}'.format(element, torch.mean(charges[element]), torch.mean(charges[f'X_{element}']), 
                                                                                        torch.mean(charges[f'J_{element}']), torch.mean(charge_percent['Es']), torch.mean(charge_percent['Ewald_tot'])))
                
                # NOTE: change the output layer bias to make the output Es as mean true_eng (deprecated: may not be general if Es is positive!!)
                # self.nn_models[element]['outputLayer'].bias.data[0] += float(torch.sum(batch.energies)/len(batch.forces) - torch.mean(charge_percent['Es'])) 
                if np.isnan(float(torch.mean(charges[f'X_{element}']))): # LOST SOME ACCURACY BUT OKAY
                    charges[f'X_{element}'] = torch.tensor([0.0]).double()
                    charges[f'J_{element}'] = torch.tensor([0.0]).double()
                
                # NOTE: change the output bias the make the output x[:, 1] to be similar as en_dict
                self.nn_models[element]['outputLayer'].bias.data[1] -= en_dict[element]*self.en_frac - float(torch.mean(charges[f'X_{element}'])) 

                if self.ifhardness:
                    self.nn_models[element]['outputLayer'].bias.data[2] -= hardness_dict[element] - float(torch.mean(charges[f'J_{element}'])) 
                
                # change the Es to fit the eV/atom scale
                if self.Es_frac != 0.0:
                    # self.nn_models[element]['outputLayer'].bias.data[0] = cohe_dict[element] - ref_const_eng * self.Es_frac 
                    self.nn_models[element]['outputLayer'].bias.data[0] = cohe_dict[element] / ref_ratio * self.Es_frac 
                    # print('{}: {:.4f} eV/atom'.format(element, cohe_dict[element] * ref_ratio))
            
            energies, forces, charges, charge_percent = self.forward(batch.allElement_fps, batch.dgdx, batch, batch_ele, 'cpu', logger=None)
            
            for element in eles:
                print('{} info after re-initialize: q: {:.2f}, X: {:.2f}, J: {:.2f}, Es: {:.2f}, Eele: {:.2f}'.format(element, torch.mean(charges[element]), torch.mean(charges[f'X_{element}']),
                                                                                        torch.mean(charges[f'J_{element}']), torch.mean(charge_percent['Es']), torch.mean(charge_percent['Ewald_tot'])))
                tuned_tag[element] = True
            
            # if all value in tuned_tag are True, break
            if all(tuned_tag.values()):
                break

        # NOTE: Debug: make nodes related to charge to be zeros
        # for element in self.elements:
        #     self.nn_models[element]['outputLayer'].weight.data[1] = torch.zeros(np.shape(self.nn_models[element]['outputLayer'].weight.data[1])).double()
        #     self.nn_models[element]['outputLayer'].weight.data[2] = torch.zeros(np.shape(self.nn_models[element]['outputLayer'].weight.data[2])).double()
        #     self.nn_models[element]['outputLayer'].bias.data[1] = torch.zeros(np.shape(self.nn_models[element]['outputLayer'].bias.data[1])).double()
        #     self.nn_models[element]['outputLayer'].bias.data[2] = torch.zeros(np.shape(self.nn_models[element]['outputLayer'].bias.data[2])).double()

        #     self.nn_models[element]['outputLayer'].bias.data[1] -= 1e-5


        return 

    """
    forward function: define the operation that acts on each layer of neural network
    fps: list of tensor of fingerprints for one image with requires_grad=True
         for each tensor, must have requires_grad=True to get gradient
         {'H':tensor([ [G1,G2,...,Gg],  Atom 1  in Image 1
                       [G1,G2,...,Gg],  Atom 2  in Image 1
                              ...
                       [G1,G2,...,Gg],  Atom N1 in Image 1
                       [G1,G2,...,Gg],  Atom 1  in Image 2
                              ...
                       [G1,G2,...,Gg],  Atom N2 in Image 2
                              ...
                       [G1,G2,...,Gg],  Atom NM in Image M
                      ])
          'Pd': ...}
    imageIndices: used to sum up energy for each image over atoms
    nimages: number of images in the training batch
    fp_d: derivative of fingeprints
          format: refer to dgdx in function 'generateInputs()'
    dEdg_AtomIndices: used to gather dEdg for forces calculations
    force_AtomIndices: used to sum up force for each atom over neighbor atoms 
    natomsPerElement: total number of atoms of each type of element.
            {'H':4,'Pd':26}
    ntotalAtoms: total number of atoms in the training batch
    """
    
    def forward(self, fps, fp_d, batch, batch_ele, device=torch.device("cpu"), logger=None, if_force=True):
    #def forward(self, batch):
        #fps = batch.allElement_fps
        #fp_d = batch.dgdx
        #for params in self.nn_models.parameters():
        #    print(params)
        #    break
        
        self.imageIndices = batch.fp_imageIndices
        self.nimages = batch.nimages

        #self.fp_d = fp_d
        self.dEdg_AtomIndices = batch.dEdg_AtomIndices

        self.force_AtomIndices = batch.force_AtomIndices
        self.force_AtomIndices = self.force_AtomIndices.to(device)

        self.natomsPerElement = batch.natomsPerElement

        self.ntotalAtoms = batch.ntotalAtoms

        # Initialize energies and columbic energies 
        energies = torch.tensor([[0.0]] * self.nimages,device=device).double()
        energies_ele = torch.zeros(self.nimages,device=device).double()
        energies_qeq = torch.zeros(self.nimages,device=device).double()
            
        # final size equal to batch.forces [self.ntotalAtoms, 3]
        forces = torch.tensor([], dtype=torch.double, device=device)
        Ele_forces1 = torch.tensor([], dtype=torch.double, device=device)
        Ele_forces2 = torch.tensor([], dtype=torch.double, device=device)

        # Alan: record electronegativity of elements(as key of dict)
        NN_X_dict = {}
        
        # Alan: record atomic hardness of elements(as key of dict)
        NN_J_dict = {}

        # Alan: record the info of mean pred_energies in ONE BATCH
        char_percent = {
            'Erecip': 0,
            'Ereal': 0,
            'Epoint': 0,
            'Echarged': 0,
            'Eqeq': 0,
            'Epercent': 0,
            'Es': 0,
            'Ewald_tot': 0,
        }

        Q_tot_lst = []
        for image_index in range(self.nimages):
            ewald_FPs = batch_ele[0][image_index]
            e = ewald_FPs.total_charge
            Q_tot_lst.append(e)
        Q_tot_lst = torch.stack(Q_tot_lst) # shape: [nimages, 1]

        # NOTE: see config.yaml for details
        input_dict = {}
        if 'none' in self.fp_timer:
            for element in self.elements:
                input_dict[element] = 1
        else:
            if self.fp_timer == 'exp':
                Q_tot_lst = torch.exp(self.fp_timer_alpha * Q_tot_lst)
            elif self.fp_timer == 'cosh':
                Q_tot_lst = torch.cosh(self.fp_timer_alpha * Q_tot_lst)
            elif self.fp_timer == 'square':
                Q_tot_lst = (self.fp_timer_alpha * Q_tot_lst) ** 2 + 1.0
            elif self.fp_timer == 'one':
                Q_tot_lst = Q_tot_lst # itself, no transformation
            for element in self.elements:
                img_ind = self.imageIndices[element].flatten()
                input_dict[element] = Q_tot_lst[img_ind] # shape: [natomsPerElement[element], 1]

        fps2 = dict()
        for element in self.elements:
            if self.fp_timer != 'one':
                fps2[element] = fps[element] * input_dict[element] # shape: [natomsPerElement[element], nFPs[element]]
            else:
                input_dict[element] = input_dict[element].expand(-1, self.fp_timer_alpha) # shape: [natomsPerElement[element], alpha]
                fps2[element] = torch.cat((fps[element], input_dict[element]), dim=1) # shape: [natomsPerElement[element], nFPs[element]+alpha]
                
        # Alan: record the accumulated number of certain element types
        chem_num_dict = {}

        for element in self.elements:
            fp = fps2[element]
            if len(fp) == 0:
                continue
            x = self.actF(self.nn_models[element]['inputLayer'](fp))
            #x = self.actF_1(self.nn_models[element]['inputLayer'](fp))
            x = x.to(device)
            if self.debug == True:
                hl_count = 0
                dead_hid = {}
                dead_hid['element'] = element
                dead_in = 0

                f_input = open('input.dat','a')
                for i in x.detach().numpy():
                    f_input.write('{}\n'.format(' '.join(map(str,i))))
                f_input.flush()
                f_input.close()
                f_input_mean = open('input_mean.dat','a')
                f_input_mean.write('{}\n'.format(' '.join(map(str,np.sum(x.detach().numpy(),axis=0)/self.ntotalAtoms))))
                f_input_mean.flush()
                f_input_mean.close()

                if self.activation == 'sigmoid':
                    dead_in += np.count_nonzero( (x.detach().numpy() < 0.001) )
                    dead_in += np.count_nonzero((0.999 < x.detach().numpy())  )
                    dead_hid[hl_count] = (dead_in, x.numel())
                    hl_count += 1
                if self.activation == 'tanh':
                    dead_in += np.count_nonzero( (x.detach().numpy() < -0.999) )
                    dead_in += np.count_nonzero((0.999 < x.detach().numpy())  )
                    dead_hid[hl_count] = (dead_in, x.numel())
                    hl_count += 1
                if self.activation == 'relu' or self.activation == 'softplus':
                    dead_in += np.count_nonzero( (x.detach().numpy() < 0.0) )
                    dead_hid[hl_count] = (dead_in, x.numel())
                    hl_count += 1

            for hd_name in self.hd_names[element]:
                dead_tmp = 0
                x = self.actF(self.nn_models[element][hd_name](x))
                #x = self.actF_2(self.nn_models[element][hd_name](x))
                x = x.to(device)
                if self.debug == True:

                    f_hid = open('{}.dat'.format(hd_name),'a')
                    for i in x.detach().numpy():
                        f_hid.write('{}\n'.format(' '.join(map(str,i))))
                    f_hid.flush()
                    f_hid.close()
                    f_hid_mean = open('{}_mean.dat'.format(hd_name),'a')
                    f_hid_mean.write('{}\n'.format(' '.join(map(str,np.sum(x.detach().numpy(),axis=0)/self.ntotalAtoms))))
                    f_hid_mean.flush()
                    f_hid_mean.close()

                    if self.activation == 'sigmoid':
                        dead_tmp += np.count_nonzero( (x.detach().numpy() < 0.001) )
                        dead_tmp += np.count_nonzero((0.999 < x.detach().numpy())  )
                    if self.activation == 'tanh':
                        dead_tmp += np.count_nonzero( (x.detach().numpy() < -0.999) )
                        dead_tmp += np.count_nonzero((0.999 < x.detach().numpy())  )
                    if self.activation == 'relu' or self.activation == 'softplus':
                        dead_in += np.count_nonzero( (x.detach().numpy() < 0.0) )
                        dead_hid[hl_count] = (dead_in, x.numel())

                    dead_hid[hl_count] = (dead_tmp, x.numel())
                    hl_count += 1

            if self.scalerType == 'MinMaxScaler':
                #x = self.actF(self.nn_models[element]['outputLayer'](x))
                x = self.nn_models[element]['outputLayer'](x)
            elif self.scalerType == 'LinearScaler':
                if self.adjust:
                    #x = self.actF(self.nn_models[element]['outputLayer'](x))
                    x = self.nn_models[element]['outputLayer'](x)
                else:
                    x = self.nn_models[element]['outputLayer'](x)
            elif self.scalerType == 'LogScaler':
                x = self.nn_models[element]['outputLayer'](x)
            elif self.scalerType == 'STDScaler':
                x = self.nn_models[element]['outputLayer'](x)
            elif self.scalerType == 'NoScaler':
                x = self.nn_models[element]['outputLayer'](x)
            #print('x', x)
            x = x.to(device)

            # benchmarking
            # x[:, 0] = x[:, 0].clone().detach()
            # x[:, 1] = x[:, 1].clone().detach()
            # x[:, 2] = x[:, 2].clone().detach()

            if self.debug == True:
                f_out = open('output.dat','a')
                for i in x.detach().numpy():
                    f_out.write('{}\n'.format(' '.join(map(str,i))))
                f_out.flush()
                f_out.close()
                f_out_mean = open('output_mean.dat','a')
                f_out_mean.write('{}\n'.format(' '.join(map(str,np.sum(x.detach().numpy(),axis=0)/self.ntotalAtoms))))
                f_out_mean.flush()
                f_out_mean.close()
                logger.info('%s', "  ".join("{}".format(v) for k,v in dead_hid.items()))

            '''
            # Alan: x[:, 0] is the NN output of the predicted energies of the input batches/
            # for certain element (that is, certain NN model)
            # **Ei**: Add the energies of same type of element in the same image together
            '''
            self.imageIndices[element] = self.imageIndices[element].to(device)

            energies_element = torch.zeros(
                self.nimages, 1, device=device).double().scatter_add_(
                    0, self.imageIndices[element], x[:, 0].view(-1, 1))
            # total energies every images: length = self.nimages
            energies = torch.add(energies, energies_element) 

            # TODO: training in batchs, it maybe better to define torch.tensor() ahead of time
            #x.backward(torch.tensor([[1.0]]*self.natomsPerElement[element]), retain_graph=True) 
            if if_force:
                dedg, = torch.autograd.grad(energies, fp, 
                                    grad_outputs=energies.data.new(energies.shape).fill_(1.0),
                                    retain_graph=True,
                                    create_graph=True,
                                    allow_unused=True)
                if self.fp_timer != 'one':
                    dedg *= input_dict[element]
                else:
                    dedg = dedg[:, :-self.fp_timer_alpha] # remove the additional dimension for timer

                fp_d[element] = fp_d[element].to(device) 
                currforces = torch.sum(torch.mul(fp_d[element],
                                        torch.flatten(dedg)[self.dEdg_AtomIndices[element]]), 1) 
                
                forces = torch.cat([forces, currforces])
            else:
                pass

            '''
            # NOTE: 
            # x[:, 0] = all atomic energy of this element type in this batch; 
            # x[:, 1] = all electronegativity of this element type in this batch
            # x[:, 2] = all atomic hardness of this element type in this batch
            '''

            # create trainable parameters a and b, x[:, 1] = x[:, 1] + a*qtot + b, 
            # where X is the electronegativity of the element
            if 'linear' in self.fp_timer:
                X_weight = self.nn_models[element]['X_weight'].val
                X_bias = self.nn_models[element]['X_bias'].val
                # J_weight = self.nn_models[element]['J_weight'].val
                # J_bias = self.nn_models[element]['J_bias'].val
            
                qtot = Q_tot_lst[self.imageIndices[element]].flatten() # shape: [natomsPerElement[element], 1]

                x[:, 1] = x[:, 1] + qtot * X_weight + X_bias
                # x[:, 2] = x[:, 2] + qtot * J_weight + J_bias # if together with X_weight and X_bias, the phi-qtot is not longer linear
            
            if 'analytical' in self.fp_timer:
                chi_qtot_linear = True
            else:
                chi_qtot_linear = False

            # NOTE: Debug, turn the charge on after nIter steps
            # if nIter < 200:
            #     x[:, 1] = torch.zeros(np.shape(x[:, 1]))
            #     x[:, 2] = torch.zeros(np.shape(x[:, 2]))
            # else:
            #     print('chg un-freezed')
            #     print(self.nn_models[element]['outputLayer'].weight.data[1])

            # NN_X_dict[element] = torch.abs(x[:, 1].view(-1, 1))
            # print('element: {}, X mean: {}, J mean: {}'.format(element, torch.mean(x[:, 1]), torch.mean(x[:, 2])))
            NN_X_dict[element] = x[:, 1].view(-1, 1) * -1
            if self.ifhardness:
                # NN_J_dict[element] = torch.abs(x[:, 2].view(-1, 1)) 
                NN_J_dict[element] = x[:, 2].view(-1, 1) * -1
            else:
                NN_J_dict[element] = torch.full_like(x[:, 1], hardness_dict[element])

            # Alan: use for define which image in 'charges' variable
            chem_num_dict[element] = 0

        # for record charge.log
        char_log_dict = dict()
        for element in self.elements:
            char_log_dict[element] = torch.tensor([], dtype=torch.double, device=device)
        for element in self.elements:
            char_log_dict['X_'+element] = torch.tensor([], dtype=torch.double, device=device)
        for element in self.elements:
            char_log_dict['J_'+element] = torch.tensor([], dtype=torch.double, device=device)


        # the sequence is the same as the acfs
        for image_index in range(self.nimages):
            ewald_FPs = batch_ele[0][image_index]

            # NOTE: update _oxi_states, input X(electronegativity) and J(atomic hardness)
            _oxi_states, _lambda, _ele_negativity, _hardness = ewald_FPs.set_oxi_states(NN_X_dict, NN_J_dict, chem_num_dict, chi_qtot_linear)

            # Update the chem_num_dict in each image ( atom numbers in elements)
            for element in self.elements:
                if element in ewald_FPs.chem_num_dict.keys():
                    chem_num_dict[element] += ewald_FPs.chem_num_dict[element]

            # NOTE: Compute the correction for a charged cell, deprecated for qeq
            # tuned_oxi_states = ewald_FPs.set_net_charge_energy()
            
            # for record charge.log
            for i, element in enumerate(ewald_FPs.chem_symbol):
                char_log_dict[element] = torch.cat([char_log_dict[element], _oxi_states[i].view(-1, 1)])
                char_log_dict['X_'+element] = torch.cat([char_log_dict['X_'+element], _ele_negativity[i].view(-1, 1)])
                char_log_dict['J_'+element] = torch.cat([char_log_dict['J_'+element], _hardness[i].view(-1, 1)])

            '''
            **Eele**: Add the energies of same type of 
            element in the same image together
            '''
            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # profile = lp(ewald_FPs._calc_recip)
            # profile()
            # profile1 = lp(ewald_FPs._calc_real_and_point)
            # profile1()
            # lp.print_stats()

            ewald_tot = ewald_FPs.total_energy
            qeq_term = ewald_FPs._calc_qeq_term(q_grad=True)

            # energies_ele[image_index] = qeq_term
            energies_ele[image_index] = ewald_tot + qeq_term 
            energies_qeq[image_index] = qeq_term

            # benchmarking, theortically dUtot/dq = 0
            # dUtotdq, = torch.autograd.grad(ewald_tot + qeq_term, ewald_FPs._oxi_states, 
            #                             retain_graph=True,
            #                             create_graph=True)
            # print(dUtotdq)

            '''
            # NOTE: dE/dr|q = -qi*qj/r2
            '''
            # curr_Eleforces1 = torch.zeros(ewald_FPs.f_recip.size())
            if if_force:
                curr_Eleforces1 = ewald_FPs.f_recip + ewald_FPs.f_real
                Ele_forces1 = torch.cat([Ele_forces1, curr_Eleforces1])
            else:
                pass

            '''
            NOTE: still debugging: if we want Etot = Eshort+Eelec, we need dEledq*dqdr
            '''
            # dqdr = ewald_FPs._calc_partialA_partialr()
            # dEledq = torch.autograd.grad(ewald_tot, ewald_FPs._oxi_states, retain_graph=True)[0] # nograd
            # curr_Eleforces1 = curr_Eleforces1 - dEledq.reshape(-1, 1)*dqdr

            '''---------------------------------'''


            char_percent['Erecip'] += ewald_FPs.reciprocal_space_energy
            char_percent['Ereal'] += ewald_FPs.real_space_energy
            char_percent['Epoint'] += ewald_FPs.point_energy
            char_percent['Eqeq'] += qeq_term
            char_percent['Echarged'] += ewald_FPs._charged_cell_energy
            
            # print('Ewald_tot = {:.7f}'.format(ewald_tot))
            # print('Ereal = {:.7f}'.format(char_percent['Ereal']))
            # print('Epoint = {:.7f}'.format(char_percent['Epoint']))
            # print('Erecip = {:.7f}'.format(char_percent['Erecip']))
            # print('Eqeq = {:.7f}'.format(char_percent['Eqeq']))
            # print('Echarged = {:.7f}'.format(char_percent['Echarged']))


        '''
        # NOTE: OPTION #1: d(E+Xqi+J*qi^2)/dq|r * dq/dg * dg/dr (q_grad=True)
                NOW I want the J(atomic hardness) also the function of g
        '''
        if if_force:
            for element in self.elements:
                fp = fps2[element]
                if len(fp) == 0:
                    continue
                deledg, = torch.autograd.grad(energies_ele, fp, 
                                            grad_outputs=energies_ele.data.new(energies_ele.shape).fill_(1.0),
                                            retain_graph=True,
                                            create_graph=True,
                                            allow_unused=True)
                if self.fp_timer != 'one':
                    deledg *= input_dict[element]
                else:
                    deledg = deledg[:, :-self.fp_timer_alpha] # remove the additional dimension for timer

                fp_d[element] = fp_d[element].to(device) 
                curr_Eleforces2 = torch.sum(torch.mul(fp_d[element],
                                            torch.flatten(deledg)[self.dEdg_AtomIndices[element]]), 1)
                Ele_forces2 = torch.cat([Ele_forces2, curr_Eleforces2]) 
        else:
            pass


        '''
        # NOTE: OPTION #2:  d(Xqi+J*qi^2)/dg|q * dg/dr, PLEASE SET ewald_FPs._calc_qeq_term(q_grad=False)
                HOWEVER, if we set q_grad = False, then dloss(Force term)/dg(dw) is not corresponded. 
                Therefore this is NOT suitable for Training, only predicting!!! (please refer SI of paper: Behler, Nat Commun 12, 398 (2021))
        '''

        # for element in self.elements:
        #     fp = fps[element]
        #     deledg, = torch.autograd.grad(energies_qeq, fp[:, :self.nFPs[element]], 
        #                                 grad_outputs=energies_qeq.data.new(energies_qeq.shape).fill_(1.0),
        #                                 retain_graph=True,
        #                                 create_graph=True)
        #     fp_d[element] = fp_d[element].to(device) 
        #     curr_Eleforces2 = torch.sum(torch.mul(fp_d[element],
        #                                 torch.flatten(deledg)[self.dEdg_AtomIndices[element]]), 1)
        #     Ele_forces2 = torch.cat([Ele_forces2, curr_Eleforces2])


        # Record 'Es', 'Ewald_tot', 'Epercent' key (no require_grad)
        Es_temp = energies.clone().detach().flatten()
        Ewald_tot_temp = energies_ele.clone().detach().flatten()
        char_percent['Epercent'] = torch.sum((Ewald_tot_temp/(Ewald_tot_temp + Es_temp))) * 100
        char_percent['Es'] = torch.sum(Es_temp) 
        char_percent['Ewald_tot'] = torch.sum(Ewald_tot_temp) 

        # take average value over this batch
        for key in char_percent.keys():
            char_percent[key] = torch.tensor([char_percent[key] / self.nimages])

        #energies = torch.flatten(torch.mul(energies, self.slope))
        #print('energies', energies)
        #print('noSumForce:', forces)
        #print("self.force_AtomIndices", self.force_AtomIndices.numpy())
        #save_data(forces, filename='forces.pl')
        #save_data(self.force_AtomIndices, filename='forceAtomInd.new')
        #print("self.ntotalAtoms", self.ntotalAtoms)
        
        #forces = torch.mul(
        #          torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
        #          scatter_add_(0, self.force_AtomIndices, forces), -self.slope) 
        
        if self.scalerType == 'MinMaxScaler':
            energies = torch.flatten(energies)
            forces = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                      scatter_add_(0, self.force_AtomIndices, forces), -self.slope)
            # add electric part
            Ele_forces1 = torch.mul(Ele_forces1, self.slope)
            Ele_forces2 = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                    scatter_add_(0, self.force_AtomIndices, Ele_forces2), -self.slope)

            energies = torch.add(energies, energies_ele)
            forces = forces + Ele_forces1 + Ele_forces2

            if self.adjust:
                energies = torch.mul(energies, self.slope)+self.intercept
            else:
                forces = (forces - self.fc_intercept)/self.slope

        elif self.scalerType == 'LinearScaler':
            energies = torch.flatten(torch.mul(energies, self.slope))
            forces = torch.mul(
                      torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                      scatter_add_(0, self.force_AtomIndices, forces), -self.slope)
            
            # add electric part
            Ele_forces1 = torch.mul(Ele_forces1, self.slope)
            Ele_forces2 = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                    scatter_add_(0, self.force_AtomIndices, Ele_forces2), -self.slope)
            energies_ele = torch.mul(energies_ele, self.slope)

            energies = torch.add(energies, energies_ele)
            forces = forces + Ele_forces1 + Ele_forces2
            
            if self.adjust:
                energies = energies + self.intercept

        elif self.scalerType == 'NoScaler':
            # If is not benchmark, should turn on both of these two flags
            if_short = self.if_short
            if_long = self.if_long

            if if_short:
                energies = torch.flatten(torch.mul(energies, self.slope))
                if if_force:
                    forces = torch.mul(
                        torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                        scatter_add_(0, self.force_AtomIndices, forces), -self.slope)
                else:
                    pass
            else:
                # if short is False, then energies and forces is 0
                energies = torch.tensor([[0.0]] * self.nimages,device=device).double()
                forces = torch.zeros(Ele_forces1.shape).double()

            if if_long:
                energies = torch.add(energies, energies_ele)
                if if_force:
                    Ele_forces1 = torch.mul(Ele_forces1, self.slope)
                    Ele_forces2 = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                            scatter_add_(0, self.force_AtomIndices, Ele_forces2), -self.slope)
                    forces = forces + Ele_forces1 + Ele_forces2
                else:
                    pass

        elif self.scalerType == 'LogScaler':
            energies = torch.flatten(energies)
            expE = torch.exp(batch.eToForces)
            forces = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                      scatter_add_(0, self.force_AtomIndices, forces), -self.eRange*expE)
            
            # add electric part
            Ele_forces1 = torch.mul(Ele_forces1, self.eRange*expE)
            Ele_forces2 = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                    scatter_add_(0, self.force_AtomIndices, Ele_forces2), -self.eRange*expE)
                    
            energies = torch.add(energies, energies_ele)
            forces = forces + Ele_forces1 + Ele_forces2

        elif self.scalerType == 'STDScaler':
            energies = torch.flatten(energies)
            forces = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                      scatter_add_(0, self.force_AtomIndices, forces), -self.ef_coef)
            
            # add electric part
            Ele_forces1 = torch.mul(Ele_forces1, self.ef_coef)
            Ele_forces2 = torch.mul(torch.zeros([self.ntotalAtoms, 3], dtype=torch.double, device=device).\
                    scatter_add_(0, self.force_AtomIndices, Ele_forces2), -self.ef_coef)

            energies = torch.add(energies, energies_ele)
            forces = forces + Ele_forces1 + Ele_forces2
        
        # for key in NN_char_dict.keys():
        #     NN_char_dict_nograd[key] = NN_char_dict[key].clone().detach()
        # print('--- Batch Summary ---')
        # for ele in self.elements:
        #     print('Mean charge of element {}: {:.4f}'.format(ele, torch.mean(char_log_dict[ele].clone().detach())))
        # for key in char_percent.keys():
        #     print('Mean {} in this batch: {:.4f} eV'.format(key, char_percent[key].item()))
        # print(' ')

        return energies, forces, char_log_dict, char_percent  #, dead_hid

    def parametersDict(self):
        Modelparameters = {}
        Modelparameters['hiddenlayers'] = self.hiddenlayers
        Modelparameters['nFPs'] = self.nFPs
        if self.scalerType in ['LinearScaler', 'STDScaler']:
            Modelparameters['slope'] = self.slope.data.item()
            Modelparameters['intercept'] = self.intercept.data.item()
        Modelparameters['forceTraining'] = self.forceTraining
        Modelparameters['activation'] = self.activation
        Modelparameters['scaler'] = self.scaler
        return Modelparameters

