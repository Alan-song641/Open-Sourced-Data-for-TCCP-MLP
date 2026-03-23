#!/usr/bin/env python3

from pyamff.config import ConfigClass
from ase.io import Trajectory
from pyamff.utilities.preprocessor import normalize, fetchProp, Scaler
from pyamff.neighborlist import NeighborLists
from pyamff.mlModels.pytorchNN import NeuralNetwork
from pyamff.mlModels.lossFunctions import SE, RMLSE, RMSE_noForce
from pyamff.fingerprints.behlerParrinello import represent_BP
from pyamff.utilities.dataPartition import DataPartitioner, partitionData
from pyamff.utilities.dataPartition import batchGenerator, batchGenerator_ele
from pyamff.utilities.logTool import setLogger, writeSysInfo
from pyamff.utilities.preprocessor import normalizeParas
from torch.multiprocessing import Process
from pyamff.training import Trainer
from pyamff.utilities import fileIO as io
from pyamff.fingerprints.fingerprints import Fingerprints
import math
import bz2file as bz2
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
#from pyamff.cross_validation_helper import k_fold_split,test_split, fetchProp_validation,test_model #test_model, k_fold_split
from pyamff.cross_validation_helper import test_model,train_test_split_new
from collections import OrderedDict
import torch.distributed as dist
import os, sys, time, glob
import torch
import torch.multiprocessing as mp
from numpy.random import randint
import numpy as np
import pickle
import random

os.system('rm batches/ fingerprints/ *.log -rf')

logger = setLogger()
charge_logger = setLogger(name='charge', logfile='charge.log')
charge_logger2 = setLogger(name='percent', logfile='percent_charge.log')

def init_processes(rank, size, thread, fn, 
                    partition, partition_Ele, testpartition, testpartition_Ele,
                    nBatchPerProc, maxEpoch, device, reportTestRMSE, 
                    masterAddr, masterPort, backend):

    # Initialize the distributed environment
    #print ('fn: ',fn) #bound method Trainer.parallelFit
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_ADDR'] = masterAddr
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    #os.environ['MASTER_ADDR'] = 'env://'
    #os.environ['MASTER_PORT'] = str(randint(low=10000, high=99999))
    #os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = masterPort
    #print('lauching', rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
#    dist.init_process_group('nccl', rank=rank, world_size=size)
    #fn(rank, size)
    fn(rank, size, thread, 
        partition, partition_Ele, 
        testpartition, testpartition_Ele, 
        nBatchPerProc, maxEpoch, 
        device, reportTestRMSE=reportTestRMSE, 
        logger=logger, charge_logger=charge_logger,charge_logger2=charge_logger2)

def init_fpProcesses(rank, size, fn, nFPs, nBatchPerProc, batchIDs, trainingImages, properties, normalize, logger,
                     fpDir, useExisting, test, chg_grad, real_space_cut, compute_forces, masterAddr, masterPort, backend):

    #Initialize the distributed environment.
    #print ("fn : ",fn) #bound method loop images
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_ADDR'] = masterAddr
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    #os.environ['MASTER_ADDR'] = 'env://'
    #os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = masterPort
    #dist.init_process_group(backend, rank=rank, world_size=size)
    dist.init_process_group('gloo', rank=rank, world_size=size)
    fn(rank, size, nFPs, nBatchPerProc, batchIDs, trainingImages, properties, 
        normalize, logger, fpDir, useExisting, test, chg_grad, real_space_cut, compute_forces)

def main():
    
    # Read in parameters

    st = time.time()
    writeSysInfo(logger)
    cwd = os.getcwd()
    logger.info('=======================================================')
    logger.info('Starting a PyAMFF job at %s', time.strftime('%X %x %Z'))
    logger.info('=======================================================')
    logger.info('Reading inputs from %s', cwd+'/config.ini')
    print('')
    print('=====================')
    print('Starting a PyAMFF job')
    print('=====================')
    print('')
    print('%8.2fs: Reading inputs' % (time.time()-st))
    config = ConfigClass()
    config.initialize()

    if config.config["use_deterministic"]:
        torch.manual_seed(0)
        random.seed(a=0)
        np.random.seed(seed=0)
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
    
    fp_paras = config.config['fp_paras'].fp_paras
    nFPs = {}
    for key in fp_paras.keys():
        nFPs[key] = len(fp_paras[key])
        #print(key, fp_paras[key])
    # print ("nFPs: ",nFPs)
    # Read in images

    useCuda = False

    if config.config['device_type'] == 'GPU' and torch.cuda.is_available():
        useCuda = True

    device = torch.device("cuda:1" if useCuda else "cpu")

    logger.info('Reading training images from %s' % cwd+'/'+config.config['trajectory_file'])

    # We also need to check if user gave us a test trajectory file
    train_test_mode = False # This will be the variable for checking
    if config.config['run_type'] == 'train_testFF':
        train_test_mode = True

    #print('%8.2fs: Reading training images' % (time.time()-st))
    if os.path.isfile(config.config['trajectory_file']):
        images = Trajectory(config.config['trajectory_file'], 'r')
    else:
        print("Trajectory file %s does not exist" % config.config['trajectory_file'], sys.stderr)
        sys.exit(2)

    if train_test_mode and os.path.isfile(config.config['test_trajectory_file']): # checking if test traj file exists
        logger.info('Reading testing images from %s' % cwd+'/'+config.config['test_trajectory_file'])
        testImages = Trajectory(config.config['test_trajectory_file'], 'r')
    else:
        if train_test_mode: # User wanted train test mode but could not find test trajectory
            print("Testing Trajectory file %s does not exist" % config.config['test_trajectory_file'], sys.stderr)
            sys.exit(2)
        else:
            train_test_mode = False # User wanted regular training.
    # Preprocess and check properties and images

    logger.info('Checking and preprocessing training images')
    #print('%8.2fs: Checking and preprocessing training images'%(time.time()-st))
    uniq_elements=config.config['fp_paras'].uniq_elements
    if config.config['use_cohesive_energy']:
        coeh = config.config['fp_paras'].refEs
        refEs = OrderedDict(zip(uniq_elements, coeh))
    else:
        refEs = None

    # Define scaler for normalization

    #print('%8.2fs: Checking for scaler:'%(time.time()-st),config.config['scaler_type'])
    scaler = Scaler(scalerType=config.config['scaler_type'], forceTraining=config.config['force_training'],  activeLearning=False, loss_type=None, cohe=config.config['use_cohesive_energy'], device=device)
    scaler = scaler.set_scaler()
    test_scaler = Scaler(scalerType=config.config['scaler_type'], forceTraining=config.config['force_training'],  activeLearning=False, loss_type=None, cohe=config.config['use_cohesive_energy'], device=device)
    test_scaler = test_scaler.set_scaler()
    if config.config['scaler_type'] in ['NoScaler', 'LinearScaler', 'MinMaxScaler','STDScaler']:
        scaler.adjust = config.config['adjust']
        test_scaler.adjust = config.config['adjust']
    if scaler is None:
        print('Normalization function is not defined!', sys.stderr)
        sys.exit()
   
    # No Cross Validation - Train 1 Model ; Normal Pyamff
    # Process training data
    #see if need to work with test images as well
    if train_test_mode:
        print('%8.2fs: Processing test data' % (time.time()-st))
        testImages, testProperties, test_scaler = fetchProp(testImages,
                                                            refEs=refEs,
                                                            scaler=test_scaler,
                                                            forceTraining=config.config['force_training'])
    
    print('%8.2fs: Processing training data' % (time.time()-st))
    trainingImages, properties, scaler = fetchProp(images, 
                                                    refEs=refEs, 
                                                    scaler=scaler,
                                                    forceTraining=config.config['force_training'])

    nImages = len(trainingImages)
    logger.info('  Number of training images: %d', nImages)
    print(' '*12,'Number of training images: %d' % nImages)
    # Ok we need to report testing images length as well
    if train_test_mode:
        ntestImages = len(testImages)
        logger.info(' Number of testing images: %d ', ntestImages)
        print(' '*12,'Number of testing images: %d' % ntestImages)

    # Get fingerprints: we need to be careful of trainFF, testFF and train_testFF tags here

    srcData = list(trainingImages.keys()) # training Images
    if train_test_mode:
        testData = list(testImages.keys()) # testing Images

    useExisting = config.config['fp_use_existing'] # for training
    testuseExisting = config.config['fp_test_use_existing'] # for testing
    if useExisting:
        logger.info('Using pre-calculated fingerprints from %s', cwd+'/'+config.config['fp_dir'])
        print('%8.2fs: Trying to load training fingerprints' % (time.time()-st))
        fpDir = config.config['fp_dir']
        #nfpFiles = len(glob.glob(os.path.join(os.getcwd(), 'fingerprints/*')))
        nfpFiles = len(glob.glob(os.path.join(fpDir, '*'))) -1
        if nfpFiles != nImages*2:
            #print("Inconsistant number of fingerprint files: %d expected but %d found" % (nImages, nfpFiles), sys.stderr)
            print("Inconsistant number of fingerprint files: %d expected but %d found" % (nImages*2, nfpFiles))
            print(' '*12,'Fingerprints were not found')
            #sys.exit(2)
            useExisting = False
        else:
            logger.info('Using pre-calculated fingerprints from %s', cwd+'/'+config.config['fp_dir'])
            print('%8.2fs: Loading precalculated training fingerprints' % (time.time()-st))
    if not useExisting:
        fpDir = None
        print('%8.2fs: Calculating training fingerprints' % (time.time()-st))
    # now same check for testing images
    if train_test_mode:
        if testuseExisting:
            logger.info('Using pre-calculated testing fingerprints from %s', cwd+'/'+config.config['test_fp_dir'])
            testfpDir = config.config['test_fp_dir']
            nfpFiles = len(glob.glob(os.path.join(testfpDir, '*'))) 
            if nfpFiles != ntestImages*2:
                print("Inconsistant number of fingerprint files: %d expected but %d found" % (ntestImages*2, nfpFiles))
                print(' '*12,'Testing fingerprints were not found')
                testuseExisting = False
            else:
                logger.info('Loading precalculated testing fingerprints from %s', cwd+'/'+config.config['test_fp_dir'])
                print('%8.2fs: Loading precalculated testing fingerprints' %(time.time()-st))
    if not testuseExisting:
        testfpDir = None

    # continue here for regular training fingerprints. We will do testing fingerprints after this

    if config.config['fp_engine'] == 'Fortran':
        fpcalc = Fingerprints(uniq_elements=config.config['fp_paras'].uniq_elements, filename=config.config['fp_parameter_file'], nfps=nFPs)
        if fpDir is None:
            fpsDir = os.getcwd() + '/fingerprints'
        else:
            fpsDir = fpDir
        if not os.path.exists(fpsDir):
            os.mkdir(fpsDir)
        nProc = config.config['process_num']
        processes = []
        test = False # I am only adding for regular pyamff and not cross validation because I dont know that code yet
        if config.config['run_type'] == 'testFF': # this is for testing the model with testing dataset
            test = True # true because letting loop images know dont write fprange.pckl file
        batches = partitionData(srcData, nProc)
        fpRange={}
        for rank in range(nProc):
                p = Process(target=init_fpProcesses,
                            args=(rank, nProc, fpcalc.loop_images, nFPs, config.config['fp_batch_num'], 
                                batches[rank], trainingImages, properties,
                                True, logger, 
                                fpsDir, useExisting, test,
                                config.config['chg_grad'],
                                config.config['real_space_cut'],
                                config.config['force_training'],
                                config.config['master_addr'],
                                config.config['master_port'],
                                config.config['backend']
                                ))
                p.start()
                processes.append(p)
        for p in processes:
            p.join()
        if test == False: # for trainFF and train_testFF; for testFF, we will do this after model is loaded
            fname = os.path.join(fpsDir, 'fprange.pckl')
            with bz2.BZ2File(fname, 'rb') as f:
                fpRange = pickle.load(f)
            fpRange, magnitudeScale, interceptScale = normalizeParas(fpRange)
        # and now we calculate fingerprints for train_test_mode: testing images figerprints
        # right now, I am not redefing the fingerprints class because I am assuming ordering of elements is same as train file
        if train_test_mode:
            if testfpDir is None:
                tfpsDir = os.getcwd() + '/test_fingerprints'
            else:
                tfpsDir = testfpDir
            if not os.path.exists(tfpsDir):
                os.mkdir(tfpsDir)
            # using the same proc_num from above
            processes = []
            test = True # make sure it does not dump to fpRange.pckl
            batches = partitionData(testData, nProc)
            # we dont need fpRange here, we will use fpRange from training test
            for rank in range(nProc):
                p = Process(target=init_fpProcesses,
                            args=(rank, nProc, fpcalc.loop_images, nFPs, config.config['fp_batch_num'],
                                    batches[rank], testImages, testProperties,
                                    True, logger, tfpsDir, testuseExisting, test,
                                    config.config['screen_rcut'],
                                    config.config['screen_k'],
                                    config.config['force_training'],
                                    config.config['master_addr'],
                                    config.config['master_port'],
                                    config.config['backend']
                                    ))

                p.start()
                processes.append(p)
            for p in processes:
                p.join()

    if config.config['run_type']=='fingerprints':
        logger.info('Fingerprint calculation done')
        print(' '*12,'Fingerprint calculation done')
        sys.exit()

    # Alan: Data Partitioning (moved to the top of NN)
    losstol = config.config['force_coefficient'] * (config.config['force_tol'] * nImages) ** 2 \
                + config.config['energy_coefficient'] * (config.config['energy_tol'] * nImages) ** 2
    logger.info('  Energy coefficient: %f', config.config['energy_coefficient'])
    logger.info('  Force coefficent:   %f', config.config['force_coefficient'])
    logger.info('  Energy tolerance:   %f', config.config['energy_tol'])
    logger.info('  Force tolerance:    %f', config.config['force_tol'])
    logger.info('  Loss tolerance:     %f', losstol)
    print(' '*12,'Energy coefficient: %f' % config.config['energy_coefficient'])
    print(' '*12,'Force coefficent:   %f' % config.config['force_coefficient'])
    print(' '*12,'Energy tolerance:   %f' % config.config['energy_tol'])
    print(' '*12,'Force tolerance:    %f' % config.config['force_tol'])
    print(' '*12,'Loss tolerance:     %f' % losstol)
    
    logger.info('Parallelization setup:')
    print(' '*12,'Parallelization setup:')
    nProc = config.config['process_num']
    nBatchPerProc = config.config['batch_num_per_proc']
    nBatches = nProc*nBatchPerProc
    logger.info('  Total number of batches:       %d', nBatches)
    logger.info('  Number of processes:           %d', nProc)
    logger.info('  Number of batches per process: %d', nBatchPerProc)
    print(' '*14,'Number of processes:           %d' % nProc)
    print(' '*14,'Total number of batches:       %d' % nBatches)
    print(' '*14,'Number of batches per process: %d' % nBatchPerProc)
    processes = []
    #partition_sizes = [1.0 / size for _ in range(size)]
    #partitions = DataPartitioner(datafilename="fps.pckl", sizes =partition_sizes, seed=1234)
    print('%8.2fs: Partitioning data' % (time.time()-st))
    partitions = DataPartitioner(srcData=srcData,
                                    fpRange=fpRange,
                                    magnitudeScale=magnitudeScale,
                                    interceptScale=interceptScale,
                                    nProc=nProc,
                                    fpDir=fpDir,
                                    nBatches=nBatches,
                                    device=device,
                                    seed=1234,
                                    st=st)
    fpRange = partitions.fpRange
    

    print('PLEASE call this function under dlossdw folder!!!')
    
    def one_step(params):

        for p in params:
            p = p.requires_grad_(True)

        # Define the NN model
        # logger.info('Defining machine-learning model')
        # print('%8.2fs: Defining machine-learning model'%(time.time()-st))
        # logger.info('  Model type: neural_network')
        # logger.info('  Model structure: %s', ' '.join([str(x) for x in config.config['hidden_layers']]))
        # print(' '*12,'Model type: neural_network')
        # print(' '*12,'Model structure: %s' % ' '.join([str(x) for x in config.config['hidden_layers']]))
        #print ('params: ',params)
        model = NeuralNetwork(
                    hiddenlayers=config.config['hidden_layers'],
                    activation=config.config['activation_function'],
                    nFPs=nFPs,
                    forceTraining=config.config['force_training'],
                    cohE = config.config['use_cohesive_energy'],
                    # TODO: load pretrained params
                    params=params,
                    scaler=scaler,
                    debug=config.config['nn_values'],
                    initial_weights =config.config['initial_weights'],
                    ifElectronegativity=config.config['ifelectronegativity'],
                    ifhardness=config.config['ifhardness'],
                    if_short=config.config['if_short'],
                    if_long=config.config['if_long'],
                    process_number=nProc,
                    partitions=partitions
                    #slope=slope,
                    #energyRange = energyRange,
                    #forceRange = forceRange
                    )

            # Define loss function
        criterion = SE(cohe=config.config['use_cohesive_energy'],
                        energyCoefficient=1.0,  
                        forceCoefficient=1.0, 
                        device=device)

        criterion_e = SE(cohe=config.config['use_cohesive_energy'],
                        energyCoefficient=1.0,  
                        forceCoefficient=0.0 , 
                        device=device)

        criterion_f = SE(cohe=config.config['use_cohesive_energy'],
                        energyCoefficient=0.0,  
                        forceCoefficient=1.0, 
                        device=device)
    

        partition = partitions.use(rank)
        partition_Ele = partitions.use_Ele(rank)
        batch_size = math.ceil(float(len(partition))/float(nBatches))
        kwargs = {'num_workers': 0, 'pin_memory': True}
        batches = torch.utils.data.DataLoader(partition,
                                            batch_size=batch_size,
                                            collate_fn=batchGenerator,
                                            shuffle=False,
                                            **kwargs)
        batches_Ele = torch.utils.data.DataLoader(partition_Ele,
                                            batch_size=batch_size,
                                            collate_fn=batchGenerator_ele,
                                            shuffle=False)

        for i, data in enumerate(zip(batches, batches_Ele)):
            # Alan: call remaked forward() function
            batch = data[0]
            batch_ele = data[1]

            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # profile = lp(model.forward)
            # predEnergies, predForces, charges, charge_percent =profile(batch.allElement_fps, batch.dgdx,
            #                             batch, batch_ele,
            #                             device, logger=logger)
            # lp.print_stats()
                                        
            predEnergies, predForces,\
                charges_dict, charge_percent = model(
                                        batch.allElement_fps, batch.dgdx,
                                        batch, batch_ele,
                                        device, logger=logger)
            
            # initialize two dicts for logging

            loss = criterion(predEnergies, predForces, batch.energies, batch.forces,
                                          natomsEnergy = batch.natomsPerImageEnergy,
                                          natomsForce = batch.natomsPerImageForce)
            loss_e = criterion_e(predEnergies, predForces, batch.energies, batch.forces,
                                          natomsEnergy = batch.natomsPerImageEnergy,
                                          natomsForce = batch.natomsPerImageForce)
            loss_f = criterion_f(predEnergies, predForces, batch.energies, batch.forces,
                                          natomsEnergy = batch.natomsPerImageEnergy,
                                          natomsForce = batch.natomsPerImageForce)
            
            lossgrads = torch.autograd.grad(loss, model.parameters(),
                                            retain_graph=True, create_graph=False)
            lossgrads_e = torch.autograd.grad(loss_e, model.parameters(),
                                            retain_graph=True, create_graph=False)
            lossgrads_f = torch.autograd.grad(loss_f, model.parameters(),
                                            retain_graph=True, create_graph=False)
            
            lossgrads_q = torch.autograd.grad(torch.sum(charges_dict['Ti']), model.parameters(),
                                            retain_graph=True, create_graph=False)

        
        model.share_memory()

        return loss, lossgrads, loss_e, lossgrads_e, loss_f, lossgrads_f, lossgrads_q, charges_dict
    
    # ------------------------------------------------------------------------
    # NOTE: for calculate d(loss)/d(weight) by 2*2 NN
    # set ifelectronegativity = False, init_model_parameters = True
    # SEE /mnt/d/Alan/PhD/pyamff_training/train_TiO2_NN_L2/dlossdw FOLDER
    """
    params =
    ['nn_models.Ti.inputLayer.weight', 
    'nn_models.Ti.inputLayer.bias', 
    'nn_models.Ti.hiddenLayer_1.weight', 
    'nn_models.Ti.hiddenLayer_1.bias', 
    'nn_models.Ti.outputLayer.weight', 
    'nn_models.Ti.outputLayer.bias',
    'nn_models.O.inputLayer.weight',
    'nn_models.O.inputLayer.bias', 
    'nn_models.O.hiddenLayer_1.weight', 
    'nn_models.O.hiddenLayer_1.bias', 
    'nn_models.O.outputLayer.weight', 
    'nn_models.O.outputLayer.bias']
    """
    def initialize_param():
        params = list()
        # # Ti: input 2 G1
        # params.append(torch.tensor([[-0.0199, -0.7241],
        #                             [ 0.1554,  0.1525]], requires_grad=True))
        # params.append(torch.tensor([ 0.7690, -1.6104], requires_grad=True))
        # # first NN hidden layer
        # params.append(torch.tensor([[-0.2388,  0.0414],
        #                             [ 1.7062, -0.8925]], requires_grad=True))
        # params.append(torch.tensor([-0.0953,  1.1711], requires_grad=True))
        # # second NN hidden layer
        # params.append(torch.tensor([[ 0.0248, -0.0359],
        #                             [ 0.3080,  0.1615]], requires_grad=True))
        # params.append(torch.tensor([-0.9970, -0.9924], requires_grad=True))

        # # O: input 2 G1
        # params.append(torch.tensor([[-0.2224,  2.7433],
        #                             [ 0.0294, -0.1505]], requires_grad=True))
        # params.append(torch.tensor([ 0.0619, -0.4459], requires_grad=True))
        # # first NN hidden layer
        # params.append(torch.tensor([[ 0.7745,  0.1278],
        #                             [-0.2276,  0.5812]], requires_grad=True))
        # params.append(torch.tensor([ 0.1127, -0.0415], requires_grad=True))
        # # second NN hidden layer
        # params.append(torch.tensor([[ 1.1762, -0.3383],
        #                             [-0.1838, -0.9613]], requires_grad=True))
        # params.append(torch.tensor([-1.8661,  1.1778], requires_grad=True))

        model_path = '../TiO2/pyamff_100.pt'
        # model_path = '../Ge/pyamff_100.pt'

        loaded = torch.load(model_path)
        
        for i in loaded['state_dict']:
            params.append(loaded['state_dict'][i])

        return params, list(loaded['state_dict'].keys())
    # ------------------------------------------------------------------------


    step_size = 1e-4
    center = 0.0
    tot_num_l = []
    tot_ana_l = []
    tot_num_e = []
    tot_ana_e = []
    tot_num_f = []
    tot_ana_f = []
    tot_num_q = []
    tot_ana_q = []

    params, _ = initialize_param()

    # tot_len = len(params) # display the full layers
    tot_len = 1 # only display the initial layers

    for which_layer in range(tot_len):
        params, _ = initialize_param()
        len_param = len(params[which_layer].flatten()) # if weight, len=4, if bias, len=2
        for which_param in range(len_param):
            numerical_loss = list()
            numerical_loss_e = list()
            numerical_loss_f = list()
            numerical_loss_q = list()
            for i in [1, -1, 0]: # left turbulence, right turbulence, original to get analytical
                params, names = initialize_param()
                params[which_layer].flatten()[which_param].data += i * step_size + center
                loss, lossgrads, loss_e, lossgrads_e, loss_f, lossgrads_f, lossgrads_q, charge_dict = one_step(params)
                
                numerical_loss.append(loss.item())
                numerical_loss_e.append(loss_e.item())
                numerical_loss_f.append(loss_f.item())
                numerical_loss_q.append(torch.sum(charge_dict['Ti']))

            num_l = (numerical_loss[0] - numerical_loss[1]) / (2*step_size)
            num_e = (numerical_loss_e[0] - numerical_loss_e[1]) / (2*step_size)
            num_f = (numerical_loss_f[0] - numerical_loss_f[1]) / (2*step_size)
            num_q = (numerical_loss_q[0] - numerical_loss_q[1]) / (2*step_size)
        
            tot_num_l.append(num_l)
            tot_ana_l.append(lossgrads[which_layer].flatten()[which_param].item())

            tot_num_e.append(num_e)
            tot_ana_e.append(lossgrads_e[which_layer].flatten()[which_param].item())

            tot_num_f.append(num_f)
            tot_ana_f.append(lossgrads_f[which_layer].flatten()[which_param].item())

            tot_num_q.append(num_q)
            tot_ana_q.append(lossgrads_q[which_layer].flatten()[which_param].item())

            print(names[which_layer], which_param)

    # print('losses:', numerical_loss)
    # print('losses_e:', numerical_loss_e)
    # print('losses_f:', numerical_loss_f)
    # print()

    # print('numerical__dloss/dw:', tot_num_l)
    # print('analytical_dloss/dw:', tot_ana_l)
    # print()

    # print('energy_numerical__dloss/dw:', tot_num_e)
    # print('energy_analytical_dloss/dw:', tot_ana_e)
    # print()

    # print('force_numerical__dloss/dw:', tot_num_f)
    # print('force_analytical_dloss/dw:', tot_ana_f)

    print('[Energy]:')
    print('         numerical       analytical      difference')
    i = 0
    for which_layer in range(tot_len):
        params, names = initialize_param()
        len_param = len(params[which_layer].flatten()) # if weight, len=4, if bias, len=2
        print(names[which_layer])
        for which_param in range(len_param):
            print('Energy: {:.5f}      {:.5f}      {:.8f}'.format(tot_num_e[i], tot_ana_e[i],  np.divide(tot_ana_e[i], tot_num_e[i])*100 ))
            i += 1
        print('')

    print('[Forces]:')
    print('         numerical       analytical      difference')
    i = 0
    for which_layer in range(tot_len):
        params, names = initialize_param()
        len_param = len(params[which_layer].flatten()) # if weight, len=4, if bias, len=2
        print(names[which_layer])
        for which_param in range(len_param):
            print('Forces: {:.5f}      {:.5f}      {:.8f}'.format(tot_num_f[i], tot_ana_f[i], np.divide(tot_ana_f[i], tot_num_f[i])*100 ))
            i += 1
        print('')

    print('[q-hat]:')
    print('         numerical       analytical      difference')
    i = 0
    for which_layer in range(tot_len):
        params, names = initialize_param()
        len_param = len(params[which_layer].flatten()) # if weight, len=4, if bias, len=2
        print(names[which_layer])
        for which_param in range(len_param):
            print('q-hat: {:.5f}      {:.5f}      {:.8f}'.format(tot_num_q[i].item(), tot_ana_q[i], np.divide(tot_ana_q[i], tot_num_q[i].item())*100 ))
            i += 1
        print('')

    os.system('rm batches/ fingerprints/ *.log -rf')

    '''
        nn_models.Ti.inputLayer.weight
        Energy: 0.01280      0.01280      100.00000014
        Energy: 0.00683      0.00683      99.99999912
        Energy: 0.03980      0.03980      100.00000047
        Energy: 0.02840      0.02840      100.00000078

        [Forces]:
                numerical       analytical      difference
        nn_models.Ti.inputLayer.weight
        Forces: -0.01468      -0.01468      100.00000057
        Forces: -0.03896      -0.03896      100.00000072
        Forces: -0.02932      -0.02932      100.00000057
        Forces: -0.07796      -0.07796      100.00000068

        [q-hat]:
                numerical       analytical      difference
        nn_models.Ti.inputLayer.weight
        q-hat: -0.01182      -0.01182      100.00000036
        q-hat: -0.00717      -0.00717      99.99999975
        q-hat: -0.03622      -0.03622      100.00000054
        q-hat: -0.02737      -0.02737      100.00000050
    
    '''

    # # Define the pyamff training 
    # logger.info('Setup PyAMFF trainer:')
    # #print('%8.2fs: Setup PyAMFF trainer:'%(time.time()-st))
    # # TODO: Alan: haven't include the config.config['force_training'] in this function!!
    # calc = Trainer(model=model,
    #                 criterion=criterion,
    #                 optimizer=config.config['optimizer_type'],
    #                 # TODO: check it can be reloaded
    #                 fpParas=config.config['fp_paras'],
    #                 energyCoefficient=config.config['energy_coefficient'],
    #                 forceCoefficient=config.config['force_coefficient'],
    #                 lossConvergence=losstol,
    #                 energyRMSEtol=config.config['energy_tol'],
    #                 forceRMSEtol=config.config['force_tol'],
    #                 lossgradtol=config.config['loss_grad_tol'],
    #                 #TODO
    #                 learningRate=config.config['learning_rate'],
    #                 model_logfile='pyamff.pt',
    #                 logmodel_interval=100,
    #                 test_loginterval=config.config['test_log_interval'],
    #                 test_scaler=test_scaler,
    #                 debug=None,
    #                 weight_decay=config.config['weight_decay'],
    #                 fpRange=fpRange,
    #                 nImages=nImages,
    #                 tnImages=tnImages,
    #                 write_final_grads =config.config['write_final_grads'])

    # # Train the NN
    # logger.info('=======================================================')
    # logger.info('Starting training')
    # head = "{:>12s} {:>14s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format('Epoch', 'LossValue', 'EnergyLoss', 'ForceLoss','EnergyRMSE', 'ForceRMSE', 'TestEnergyRMSE', 'TestForceRMSE')
    # logger.info('%s', head)

    # #for node in range(2):
    # #  print(mastaddr[node], mastips[node])
    # #  for rank in range(int(nProc/2)):

    # print("%8.2fs: Training started" % (time.time()-st))
    # print(' '*12,'Epoch    LossValue   EnergyRMSE    ForceRMSE    TestEnergyRMSE    TestForceRMSE ')
    # train_st = time.time()
    # mp.set_start_method(config.config['mp_start_method'], force=True)
    # for rank in range(nProc):
    # #      local_rank = rank
    # #      rank = ranks_per_node * node + local_rank
    #         #print('rank', rank)

    # #      mp.spawn(calc.parallelFit, nprocs=process_number, args=(process_number, config.config['thread_num'],
    # #                                                                 partitions.use(rank),
    # #                                                                 batch_number, config.config['epochs_max'],
    # #                                                                 device, logger),join=True)
    #         p = Process(target=init_processes,
    #                     args=(rank, nProc, config.config['thread_num'],
    #                         #config.config['master_addr'], 
    #                         #mastips[node],
    #                         #config.config['master_port'],
    #                         calc.parallelFit, 
    #                         partitions.use(rank), 
    #                         partitions.use_Ele(rank),
    #                         testpartitions.use(rank),
    #                         testpartitions.use_Ele(rank),
    #                         #config.config['batch_number'],
    #                         nBatches,
    #                         config.config['epochs_max'],
    #                         device,
    #                         reportTestRMSE,
    #                         config.config['master_addr'],
    #                         config.config['master_port'],
    #                         config.config['backend']
    #                         #'nccl'
    #                         ))
    #         p.start()
    #         processes.append(p)
    # for p in processes:
    #     p.join()
    # et = time.time()
    # print("%8.2fs: Training done, time used: %.2fs" % (et-st,et-train_st))

if __name__ == "__main__":
    main()
