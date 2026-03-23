#!/usr/bin/env python
import time, os
import sys, math
import torch
import numpy as np
from collections import OrderedDict
from pyamff.utilities.dataPartition import batchGenerator, batchGenerator_ele
from pyamff.utilities.preprocessor import normalize, fetchProp
from pyamff.utilities import fileIO as io
from pyamff.mlModels.lossFunctions import LossFunction, calc_mse
from pyamff.mlModels.pytorchNN import NeuralNetwork
from pyamff.utilities.analyze import plotGradFlow
from pyamff.optimizer.lbfgs import LBFGSScipy
from pyamff.optimizer.lbfgsNew import LBFGSNew
from pyamff.optimizer.sd import SD
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pyamff.utilities.logTool import setLogger
import psutil

def print_memory_info():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30  # Convert to GB
    print('Memory usage of current process: %.4f GB' % memory_use)

def averageGradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        # Ensure all processes participate in all_reduce, even if grad is None
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        else:
            # Create a zero tensor with the same shape and dtype as param
            zero_grad = torch.zeros_like(param.data)
            dist.all_reduce(zero_grad, op=dist.ReduceOp.SUM)


class Trainer():

    def __init__(self, model, 
                 criterion=None,
                 fpParas=None,
                 energyCoefficient=1.00,
                 forceCoefficient=0.1,
                 optimizer='LBFGS',
                 learningRate=0.1,
                 lossConvergence=1e-4,
                 energyRMSEtol=0.01,
                 forceRMSEtol=0.1,
                 lossgradtol=1e-09,
                 model_logfile='pyamff.pt',
                 logmodel_interval=100,
                 test_loginterval=10,
                 test_scaler=None,
                 debug='force', 
                 weight_decay=0,
                 fpRange=None,
                 nImages=None,
                 tnImages=None,
                 write_final_grads=False):
        self.model = model
        self.criterion = criterion
        self.fpParas = fpParas
        self.energyCoefficient = energyCoefficient
        self.forceCoefficient = forceCoefficient
        self.optimizer = optimizer
        self.learningRate = learningRate
        self.lossConvergence = lossConvergence
        self.lossgradtol = lossgradtol
        self.model_logfile = model_logfile
        self.logmodel_interval = logmodel_interval
        self.test_loginterval = test_loginterval
        self.weight_decay = weight_decay
        self.debug = debug
        self.fpRange = fpRange
        self.nImages = nImages
        self.tnImages = tnImages
        self.scaler = model.scaler
        self.scalerType = model.scalerType

        # Alan: to use pyamff_test.pt in aseCalc for test set!
        self.test_model_logfile = 'pyamff_test.pt'
        self.test_scaler = test_scaler
        # self.test_scalerType = test_scaler.scalerType

        #if self.scalerType in ['MinMaxScaler','LogScaler']:
        #   self.eRange = self.model.eMinMax[1] - self.model.eMinMax[0]
        #   self.minE = self.model.eMinMax[0]
        #   self.fRange = self.model.fMinMax[1] - self.model.fMinMax[0]
        #   self.minF = self.model.fMinMax[0]
        # Variable used to store training info
        self.energyRMSE = None
        self.forceRMSE = None
        self.energyloss = 0.
        self.forceloss = 0.
        self.energyRMSEtol = energyRMSEtol
        self.forceRMSEtol = forceRMSEtol
        self.nIter = 0
        self.running = True
        self.write_final_grads = write_final_grads

        # Alan: record the change of partial charges & energies info
        self.new_logger_times = 0
        

    """
    trainingimages: {'hashedkey':Atoms(), ...}
    descriptor: amp.descriptor object. 
               Use descriptor.fingerprints to fetch fingerprint data with
               following structure:
               {'hashedkey': fps for image 1, ...}
               atomSymbols, fpdata = zip(*fps1):
               atomSymbols: a tuple ('Au', 'Au',...,)
               fpdata: a tuple ([G1, G2, ...,] for atom 1, [G1, G2, ...,] for atom 2, ...)
               Use descriptor.fingerprintprimes to fetch corresponding derivatives
    """

    # Alan: record the change of partial charges & energies info
    # TODO: output the charge and percent_charge of the test set as the function of Epoch
    def update_logger(self, log, log_dict):
        keys = log_dict.keys()
        # call two times this function means two charge_logger initialized
        if self.new_logger_times != 2: 
            log.info('Starting training')
            lines = '       Epoch  '
            for ele in keys:
                lines += '  {}  '.format(ele)
            log.info(lines)
            self.new_logger_times += 1
        
        lines = '{:12d}  '.format(self.nIter)
        for key in keys:
            lines += '{:.6f}    '.format(log_dict[key].item())
        log.info(lines)


    def saveModel(self):
        self.model_logfile = 'pyamff_{}.pt'.format(self.nIter)

        state_dict = self.model.state_dict()
        modelParameters = self.model.parametersDict()
        self.preprocessParas = {}
        if self.scalerType in ['LinearScaler', 'MinMaxScaler', 'STDScaler']:
            #Jiyoung edited: currently, only two scalers work with adjust and scaler.intercept
            #scaler.slope doesn't change 
            if self.model.adjust: #if training slopes and intercepts
                self.preprocessParas['intercept'] = self.model.intercept
                self.preprocessParas['slope'] = self.model.slope
            else: #not training slopes and intercepts
                self.preprocessParas['intercept'] = self.scaler.intercept
                self.preprocessParas['slope'] = self.scaler.slope
        if self.scalerType == 'NoScaler':
            self.preprocessParas['intercept'] = self.scaler.intercept
            self.preprocessParas['slope'] = self.scaler.slope
        if self.scalerType in ['LogScaler']:
            self.preprocessParas['energyRange'] = self.model.eMinMax
            #self.preprocessParas['forceRange'] = self.model.forceRange
        self.preprocessParas['fpRange']=self.fpRange
        self.preprocessParas['fingerprints']=self.fpParas
        torch.save({
             'state_dict': state_dict,
             'preprocessParas': self.preprocessParas,
             'Modelparameters': modelParameters},
             self.model_logfile)
        # as original 
        torch.save({
             'state_dict': state_dict,
             'preprocessParas': self.preprocessParas,
             'Modelparameters': modelParameters},
             'pyamff.pt')
        
        # if self.reportTestRMSE:
        #     self.saveTestModel()
    

    # def saveTestModel(self):
    #     self.test_model_logfile = 'pyamff_test_{}.pt'.format(self.nIter)

    #     state_dict = self.model.state_dict()
    #     modelParameters = self.model.parametersDict()
    #     modelParameters['scaler'] = self.test_scaler
    #     if self.scalerType in ['LinearScaler', 'STDScaler']:
    #         modelParameters['slope'] = self.test_scaler.slope
    #         modelParameters['intercept'] = self.test_scaler.intercept
    #     self.test_preprocessParas = {}
    #     if self.scalerType in ['LinearScaler', 'MinMaxScaler', 'STDScaler']:
    #        #Jiyoung edited: currently, only two scalers work with adjust and scaler.intercept
    #        #scaler.slope doesn't change 
    #         self.test_preprocessParas['intercept'] = self.test_scaler.intercept
    #         self.test_preprocessParas['slope'] = self.test_scaler.slope
    #     if self.scalerType == 'NoScaler':
    #        self.test_preprocessParas['intercept'] = self.test_scaler.intercept
    #        self.test_preprocessParas['slope'] = self.test_scaler.slope
    #     if self.scalerType in ['LogScaler']:
    #         #TODO: currently not working for test set!!
    #        self.test_preprocessParas['energyRange'] = self.model.eMinMax
    #        #self.preprocessParas['forceRange'] = self.model.forceRange
    #     self.test_preprocessParas['fpRange']=self.fpRange
    #     self.test_preprocessParas['fingerprints']=self.fpParas
    #     torch.save({
    #          'state_dict': state_dict,
    #          'preprocessParas': self.test_preprocessParas,
    #          'Modelparameters': modelParameters},
    #          self.test_model_logfile)



    def testRMSE(self, batches, batches_Ele, device, logger, rank, parallel=True): # function to report testRMSE during train_testFF mode

        tenergyloss = 0.
        tforceloss = 0.
        tenergyMSE = 0.
        tforceMSE = 0.
        tenergylossRMSE = 0.
        tforcelossRMSE = 0.
        tloss = 0.

        # Alan: use slope and intercept of test_scaler when calculate testRMSE
        self.ddp_model.module.scaler = self.test_scaler
        
        for i, data in enumerate(zip(batches, batches_Ele)):
            # Alan: call remaked forward() function
            batch = data[0]
            batch_ele = data[1]

            predEnergies,  predForces, \
            pred_charges, charge_percent = self.ddp_model(
                batch.allElement_fps, batch.dgdx,
                batch, batch_ele,
                device,  logger=logger)
                
            tloss += self.criterion(predEnergies, predForces, batch.energies, batch.forces,
                                  natomsEnergy = batch.natomsPerImageEnergy,
                                  natomsForce = batch.natomsPerImageForce)
                    
            tenergyloss += self.criterion.energyloss
            tforceloss += self.criterion.forceloss

            tenergyMSE_new, tforceMSE_new = self.test_scaler.calculate_mse(predEnergies, predForces, batch.energies, batch.forces,
                                                                    batch.natomsPerImageEnergy,
                                                                    batch.natomsPerImageForce)
            tenergyMSE += tenergyMSE_new
            tforceMSE  += tforceMSE_new
        
        # switch back, do not know if it is the best way
        self.ddp_model.module.scaler = self.scaler

        if math.isnan(tenergyloss.item()):
            raise ValueError('energy RMSE is nan')
        if math.isnan(tforceloss.item()):
            raise ValueError('force RMSE is nan')

        if parallel:
            dist.all_reduce(tloss, dist.ReduceOp.SUM)
            dist.all_reduce(tenergyloss, dist.ReduceOp.SUM)
            dist.all_reduce(tforceloss, dist.ReduceOp.SUM)
            if self.criterion.loss_type == 'SE':
                tloss = torch.pow(torch.div(tloss, self.tnImages), 0.5)

            if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler', 'NoScaler']:
                dist.all_reduce(tenergyMSE, dist.ReduceOp.SUM)
                dist.all_reduce(tforceMSE, dist.ReduceOp.SUM)
        tenergylossRMSE = np.sqrt(tenergyloss.item()/self.tnImages)
        tforcelossRMSE = np.sqrt(tforceloss.item()/self.tnImages)

        if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler', 'NoScaler']:
            tenergyRMSE = np.sqrt(tenergyMSE.item()/self.tnImages)
            tforceRMSE = np.sqrt(tforceMSE.item()/self.tnImages)
        else:
            tenergyRMSE = tenergylossRMSE
            tforceRMSE  = tforcelossRMSE

        self.energyRMSE = tenergyRMSE
        self.forceRMSE = tforceRMSE
        
        return tenergyRMSE, tforceRMSE


    def getLoss(self, batches, batches_Ele, device, logger, parallel=True):
        '''
        Alan: this is for optimizers except LBFGS
        NOTE: should output the charges for updating charge.log and charge_percent.log
        '''
        curr_loss = 0.
        energyloss = 0.
        forceloss = 0.
        energyMSE = 0.
        forceMSE = 0.
        energylossRMSE = 0.
        forcelossRMSE = 0.
        loss = 0.
        num = 0
        cum_loss = 0. #this will collect loss from each batch but will not be used for lossgrads to save time
        time.sleep(1.0)

        # Alan: update charge.log & percent_charge.log
        charges_epoch = dict()
        charge_percent_epoch = dict()

        for i, data in enumerate(zip(batches, batches_Ele)):
            # Alan: call remaked forward() function
            batch = data[0]
            batch_ele = data[1]

            predEnergies, predForces, \
            charges, charge_percent = self.ddp_model(
                batch.allElement_fps, batch.dgdx,
                batch, batch_ele,
                device,  logger=logger)
            
            if i == 0:
                for ele in charges.keys():
                    charges_epoch[ele] = charges[ele]
                for type in charge_percent.keys():
                    charge_percent_epoch[type] = charge_percent[type]
            else:
                for ele in charges.keys():
                    charges_epoch[ele] = torch.cat((charges_epoch[ele], charges[ele]))
                for type in charge_percent.keys():
                    charge_percent_epoch[type] = torch.cat((charge_percent_epoch[type], charge_percent[type]))

            loss = self.criterion(predEnergies, predForces, batch.energies, batch.forces,
                              natomsEnergy = batch.natomsPerImageEnergy,
                              natomsForce = batch.natomsPerImageForce)

            #print ('+140 loss before loss grad: ',loss)
            lossgrads = torch.autograd.grad(loss, self.model.parameters(),
                                            retain_graph=True, create_graph=False, allow_unused=True)

            # print('lossgrads: {}'.format(lossgrads))
            # print('loss: {}'.format(loss))

            cum_loss += loss.clone().detach() #accumulate loss for all batches
            for p, g in zip(self.model.parameters(), lossgrads):
                if num == 0:
                    p.grad = g
                else:
                    p.grad += g

            num += 1  # this counter checks if its the first batch, then assign g otherwise accumulate g

            energyloss += self.criterion.energyloss
            forceloss += self.criterion.forceloss

            energyMSE_new, forceMSE_new = self.scaler.calculate_mse(predEnergies, predForces, batch.energies, batch.forces,
                                                                    batch.natomsPerImageEnergy,
                                                                    batch.natomsPerImageForce)
            energyMSE += energyMSE_new
            forceMSE += forceMSE_new

        if parallel:
            averageGradients(self.ddp_model)
            dist.all_reduce(cum_loss, dist.ReduceOp.SUM) # all reduce should be on cum_loss and not loss
            dist.all_reduce(energyloss, dist.ReduceOp.SUM)
            dist.all_reduce(forceloss, dist.ReduceOp.SUM)
            if self.criterion.loss_type == 'SE':
                cum_loss = torch.pow(torch.div(cum_loss, self.nImages), 0.5)

            if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler','NoScaler']:
                dist.all_reduce(energyMSE, dist.ReduceOp.SUM)
                dist.all_reduce(forceMSE, dist.ReduceOp.SUM)
        energylossRMSE = np.sqrt(energyloss.item()/self.nImages)
        forcelossRMSE = np.sqrt(forceloss.item()/self.nImages)

        # update charge.log & percent_charge.log
        # Alan: averaged over batches
        # for ele in charges_epoch.keys():
        #     charges_epoch[ele] = torch.mean(charges_epoch[ele])
        # for type in charge_percent_epoch.keys():
        #     charge_percent_epoch[type] = torch.mean(charge_percent_epoch[type])

        if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler','NoScaler']:
            energyRMSE = np.sqrt(energyMSE.item()/self.nImages)
            forceRMSE = np.sqrt(forceMSE.item()/self.nImages)
        else:
            energyRMSE = energylossRMSE
            forceRMSE = forcelossRMSE

        self.energyRMSE = energyRMSE
        self.forceRMSE = forceRMSE
        #print ('getLoss loss+183 training: ',float(cum_loss.item()))
        return float(cum_loss.item()), energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE, lossgrads, charges_epoch, charge_percent_epoch



    def parallelFit(self, rank, size, thread, 
                    partition, partition_Ele, 
                    testpartition, testpartition_Ele, 
                    batch_number, maxEpochs, 
                    device, reportTestRMSE=False, 
                    logger=None, parallel=True, 
                    charge_logger=None, charge_logger2=None):

        #torch.manual_seed(0)
        #random.seed(a=0)
        #np.random.seed(seed=0)
        #torch.use_deterministic_algorithms(mode=True, warn_only=True)
        
        #os.environ['MASTER_ADDR'] = '127.0.0.1'
        #os.environ['MASTER_PORT'] = '1234'

        #dist.init_process_group('gloo', init_method='env://', rank=rank, world_size=size)
        self.running = True
        if self.optimizer == 'LBFGSScipy':
            pass
        else:
            self.model = self.model.to(device)

        if device == torch.device("cuda:0"):
            self.ddp_model = DDP(self.model, device_ids=[rank])
            n_gpus = torch.cuda.device_count()
        else:
            self.ddp_model = DDP(self.model, device_ids=[])

        torch.manual_seed(1234)
        torch.set_num_threads(thread)
        self.reportTestRMSE = reportTestRMSE
        if parallel:
            batch_size = math.ceil(float(len(partition))/float(batch_number))
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
            if reportTestRMSE:
                batch_size = math.ceil(float(len(testpartition))/float(batch_number))
                kwargs = {'num_workers':0, 'pin_memory':True}
                testBatches = torch.utils.data.DataLoader(testpartition,
                                                        batch_size=batch_size,
                                                        collate_fn=batchGenerator,
                                                        shuffle=False,
                                                        **kwargs)
                testBatches_Ele = torch.utils.data.DataLoader(testpartition_Ele,
                                                        batch_size=batch_size,
                                                        collate_fn=batchGenerator_ele,
                                                        shuffle=False,
                                                        **kwargs)
        else: # I dont know why do we have this else check
            batches = [partition]
            batches_Ele = [partition_Ele]
            testBatches = [testpartition]
            testBatches_Ele = [testpartition_Ele]

        #calculate loss and RMSE for initialized model
        #loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE = self.getLoss(batches, device, logger)
        #if rank == 0:
        #    logger.info('%s', "Initial Loss:  {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}".format(loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE))

        # Define Optimizers

        if 'LBFGS' in self.optimizer:
            # Alan: Ewald summation function currently not used for LBFGSScipy
            if self.optimizer == 'LBFGSScipy':

                optimizer = LBFGSScipy(self.model.parameters(), max_iter=maxEpochs, logger=logger, rank=rank)
                curr_loss = 0.
                energyloss = 0.
                forceloss = 0.
                energylossRMSE = 0.
                forcelossRMSE = 0.
                start_time = time.time()
                activation = {} 
                
                self.nIter = 0

                def closure():
                    loss = 0
                    energyloss = 0.
                    forceloss = 0.
                    energyMSE = 0.
                    forceMSE = 0.
                    energylossRMSE = 0.
                    forcelossRMSE = 0.
                    batchid = 0

                    # Alan: the update of optimizer._niter is in lbfgs.py
                    # self.nIter = optimizer._niter
                    

                    # Alan: update charge.log & percent_charge.log
                    charges_epoch = dict()
                    charge_percent_epoch = dict()

                    for i, data in enumerate(zip(batches, batches_Ele)):
                        # Alan: call remaked forward() function
                        batch = data[0]
                        batch_ele = data[1]

                        predEnergies, predForces,\
                         charges, charge_percent = self.model(
                                                    batch.allElement_fps, batch.dgdx,
                                                    batch, batch_ele,
                                                    device, logger=logger)
                        
                        # initialize two dicts for logging
                        if i == 0:
                            for ele in charges.keys():
                                charges_epoch[ele] = charges[ele]
                            for type in charge_percent.keys():
                                charge_percent_epoch[type] = charge_percent[type]
                        else:
                            for ele in charges.keys():
                                charges_epoch[ele] = torch.cat((charges_epoch[ele], charges[ele]))
                            for type in charge_percent.keys():
                                charge_percent_epoch[type] = torch.cat((charge_percent_epoch[type], charge_percent[type]))
                        
                        loss += self.criterion(predEnergies, predForces, batch.energies, batch.forces,
                                               natomsEnergy = batch.natomsPerImageEnergy,
                                               natomsForce = batch.natomsPerImageForce)
                        
                        lossgrads = torch.autograd.grad(loss, self.model.parameters(),
                                                        retain_graph=True, create_graph=False, allow_unused=True)

                        for p, g in zip(self.model.parameters(), lossgrads):
                            p.grad = g
                        batchid += 1
                        energyloss += self.criterion.energyloss
                        forceloss += self.criterion.forceloss
                        # Calc RMSE
                        energyMSE_new, forceMSE_new = self.scaler.calculate_mse(predEnergies, predForces, batch.energies, batch.forces,
                                                                                batch.natomsPerImageEnergy,
                                                                                batch.natomsPerImageForce)
                        energyMSE += energyMSE_new
                        forceMSE += forceMSE_new

                    if parallel:
                        averageGradients(self.model)
                        dist.all_reduce(loss, dist.ReduceOp.SUM)
                        dist.all_reduce(energyloss, dist.ReduceOp.SUM)
                        dist.all_reduce(forceloss, dist.ReduceOp.SUM)
                        if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler','NoScaler']:
                            dist.all_reduce(energyMSE, dist.ReduceOp.SUM)
                            dist.all_reduce(forceMSE, dist.ReduceOp.SUM)
                        if self.criterion.loss_type == 'SE': #this cum_loss is only for logging.
                            # unlike LBFGS torch, here loss is cum_loss
                            cum_loss = torch.pow(torch.div(loss.clone().detach(), self.nImages), 0.5)

                    energylossRMSE = np.sqrt(energyloss.item()/self.nImages)
                    forcelossRMSE = np.sqrt(forceloss.item()/self.nImages)

                    if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler','NoScaler']:
                        energyRMSE = np.sqrt(energyMSE.item()/self.nImages)
                        forceRMSE = np.sqrt(forceMSE.item()/self.nImages)
                    else:
                        energyRMSE = energylossRMSE
                        forceRMSE  = forcelossRMSE

                    # Raise error if energy RMSE or/and force RMSE is nan
                    if math.isnan(energyRMSE):
                        raise ValueError('Energy RMSE is nan')
                    if math.isnan(forceRMSE):
                        raise ValueError('Force RMSE is nan')

                    self.energyRMSE = energyRMSE
                    self.forceRMSE = forceRMSE

                    # check if testRMSE has to be calculated
                    if reportTestRMSE and self.nIter % self.test_loginterval == 0:
                        teRMSE, tfRMSE = self.testRMSE(testBatches, testBatches_Ele, device, logger, rank)
                    else: 
                        teRMSE, tfRMSE = False, False
                    if rank == 0:
                        if self.nIter < 0: #Naman: again, if you make it equal to zero, it wont report anything
                            pass
                        else:
                            # update charge.log & percent_charge.log
                            # Alan: averaged over batches
                            for ele in charges_epoch.keys():
                                charges_epoch[ele] = torch.mean(charges_epoch[ele])
                            for type in charge_percent_epoch.keys():
                                charge_percent_epoch[type] = torch.mean(charge_percent_epoch[type])
                            
                            self.update_logger(charge_logger, charges_epoch)
                            self.update_logger(charge_logger2, charge_percent_epoch)
                            if teRMSE:
                                logger.info('%s', "{:12d} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}".format(self.nIter, cum_loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE, teRMSE, tfRMSE))
                                print('%12d %12.6f %12.6f %12.6f %12.6f %12.6f' % (self.nIter, cum_loss, energyRMSE, forceRMSE, teRMSE, tfRMSE))
                            else:
                                logger.info('%s', "{:12d} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}".format(self.nIter, cum_loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE))
                                print('%12d %12.6f %12.6f %12.6f' % (self.nIter, cum_loss, energyRMSE, forceRMSE))
                    
                    self.nIter += 1
                    # add a condition to stop the training if the loss is too high
                    # if np.abs(cum_loss.item()) > 1000000 and self.nIter > 200:
                    #     self.running = False
                    #     if rank == 0:
                    #         print(' '*12, 'loss is too high, stopping training')

                    # Alan: output gradients to file
                    if rank == 0 and self.nIter % 100 == 0:
                        with open('output_grad.txt', 'w') as f:
                            for name, param in self.ddp_model.named_parameters():
                                if param.grad is not None:
                                    grad_data = param.grad.data.cpu().numpy()
                                    f.write(f"Layer: {name}\n")
                                    np.savetxt(f, grad_data.flatten(), fmt='%1.6f')
                                    f.write("\n\n")

                    if self.nIter % self.logmodel_interval == 0:
                        # while saving the model, only make rank 0 do it
                        if rank == 0:
                            self.saveModel()
                            io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")

                    if energyRMSE < self.energyRMSEtol and forceRMSE < self.forceRMSEtol:
                        logger.info('Minimization converged')
                        self.saveModel()
                        io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                        sys.exit()
                        
                    if self.nIter >= maxEpochs: #we don't want the log file to have both Minimizatino Converged and Max Epoch Reached
                        if rank == 0:
                            # logger.info('Max Epoch Reached') # TODO: the max epoch criterion is based on step function
                            print(' '*12,'Max Epoch Reached')
                            self.saveModel()
                            io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                            sys.exit()
                        
                    return loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE

                optimizer.step(closure)
                self.saveModel()
                io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                
                sys.exit()

            elif self.optimizer == 'LBFGS':
                """
                    Uses torch LBFGS optimizer. Loss is not averaged over batches.
                """
                optimizer = torch.optim.LBFGS(
                                              self.model.parameters(),
                                              lr=self.learningRate, #learning rate
                                              tolerance_grad=1e-10,
                                              tolerance_change=1e-10,
                                              #max_iter=maxEpochs,
                                              max_iter=1,
                                              #max_eval=None,
                                              max_eval=None,
                                              history_size=10,
                                              line_search_fn=None # either 'strong_wolfe' or None
                                             )

                def closure():
                    curr_loss = 0.
                    energyloss = 0.
                    forceloss = 0.
                    energyMSE = 0.
                    forceMSE = 0.
                    energylossRMSE = 0.
                    forcelossRMSE = 0.
                    loss = 0.
                    num = 0
                    cum_loss = 0. # this will accumulate the loss while loss will be used for lossgrads
                    optimizer.zero_grad()
                    time.sleep(1.0) # Naman: oh my god i hate this patch
                    
                    # Alan: update charge.log & percent_charge.log
                    charges_epoch = dict()
                    charge_percent_epoch = dict()

                    for i, data in enumerate(zip(batches, batches_Ele)):
                        # Alan: call remaked forward() function
                        batch = data[0]
                        batch_ele = data[1]

                        predEnergies, predForces,\
                         charges, charge_percent = self.ddp_model(
                                                    batch.allElement_fps, batch.dgdx,
                                                    batch, batch_ele, device, 
                                                    logger=logger)
                        # print(rank, i, predEnergies)
                        # initialize two dicts for logging
                        if i == 0:
                            for ele in charges.keys():
                                charges_epoch[ele] = charges[ele]
                            for type in charge_percent.keys():
                                charge_percent_epoch[type] = charge_percent[type]
                        else:
                            for ele in charges.keys():
                                charges_epoch[ele] = torch.cat((charges_epoch[ele], charges[ele]))
                            for type in charge_percent.keys():
                                charge_percent_epoch[type] = torch.cat((charge_percent_epoch[type], charge_percent[type]))

                        loss = self.criterion(predEnergies, predForces, batch.energies, batch.forces,
                                          natomsEnergy = batch.natomsPerImageEnergy,
                                          natomsForce = batch.natomsPerImageForce)
                        cum_loss += loss.clone().detach()
                        lossgrads = torch.autograd.grad(loss, self.model.parameters(),
                                                        retain_graph=True, create_graph=False, allow_unused=True)

                        for p, g in zip(self.model.parameters(), lossgrads):
                            if num == 0 or p.grad is None:
                                p.grad = g
                            elif g is not None:
                                p.grad += g
                        num += 1 #this is not a pretty solution, and we should think of a better way.
                        energyloss += self.criterion.energyloss
                        forceloss += self.criterion.forceloss

                        # Calc RMSE
                        energyMSE_new, forceMSE_new = self.scaler.calculate_mse(
                            predEnergies, predForces, batch.energies,
                            batch.forces, batch.natomsPerImageEnergy,
                            batch.natomsPerImageForce)
                        energyMSE += energyMSE_new
                        forceMSE  += forceMSE_new
                        if math.isnan(energyloss.item()):
                            raise ValueError('energy SE is nan')
                        if math.isnan(forceloss.item()):
                            raise ValueError('force SE is nan')
                    if parallel:
                        averageGradients(self.ddp_model)
                        dist.all_reduce(cum_loss, dist.ReduceOp.SUM)
                        dist.all_reduce(energyloss, dist.ReduceOp.SUM)
                        dist.all_reduce(forceloss, dist.ReduceOp.SUM)

                        # Alan: output gradients to file
                        if rank == 0 and self.nIter % 100 == 0:
                            with open('output_grad.txt', 'w') as f:
                                for name, param in self.ddp_model.named_parameters():
                                    if param.grad is not None:
                                        grad_data = param.grad.data.cpu().numpy()
                                        f.write(f"Layer: {name}\n")
                                        np.savetxt(f, grad_data.flatten(), fmt='%1.6f')
                                        f.write("\n\n")
                            self.saveModel()

                        if self.criterion.loss_type == 'SE': #this cum_loss is only for logging.
                            cum_loss = torch.pow(torch.div(cum_loss, self.nImages), 0.5)

                        if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler', 'NoScaler']:
                            dist.all_reduce(energyMSE, dist.ReduceOp.SUM)
                            dist.all_reduce(forceMSE, dist.ReduceOp.SUM)

                    energylossRMSE = np.sqrt(energyloss.item()/self.nImages)
                    forcelossRMSE = np.sqrt(forceloss.item()/self.nImages)

                    if self.criterion.loss_type != 'SE' or self.scalerType not in ['LinearScaler', 'NoScaler']:
                        energyRMSE = np.sqrt(energyMSE.item()/self.nImages)
                        forceRMSE = np.sqrt(forceMSE.item()/self.nImages)
                    else:
                       energyRMSE = energylossRMSE
                       forceRMSE  = forcelossRMSE

                    self.energyRMSE = energyRMSE
                    self.forceRMSE = forceRMSE

                    # check if testRMSE has to be calculated
                    if reportTestRMSE and self.nIter % self.test_loginterval == 0:
                        teRMSE, tfRMSE = self.testRMSE(testBatches, testBatches_Ele, device, logger, rank)
                    else: 
                        teRMSE, tfRMSE = False, False
                    if rank == 0:
                        if self.nIter < 0: #Naman: again, if you make it equal to zero, it wont report anything
                            pass
                        else:
                            # update charge.log & percent_charge.log
                            # Alan: averaged over batches
                            for ele in charges_epoch.keys():
                                charges_epoch[ele] = torch.mean(charges_epoch[ele])
                            for type in charge_percent_epoch.keys():
                                charge_percent_epoch[type] = torch.mean(charge_percent_epoch[type])
                            
                            self.update_logger(charge_logger, charges_epoch)
                            self.update_logger(charge_logger2, charge_percent_epoch)
                            if teRMSE:
                                logger.info('%s', "{:12d} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}".format(self.nIter, cum_loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE, teRMSE, tfRMSE))
                                print('%12d %12.6f %12.6f %12.6f %12.6f %12.6f' % (self.nIter, cum_loss, energyRMSE, forceRMSE, teRMSE, tfRMSE))
                            else:
                                logger.info('%s', "{:12d} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}".format(self.nIter, cum_loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE))
                                print('%12d %12.6f %12.6f %12.6f' % (self.nIter, cum_loss, energyRMSE, forceRMSE))
                    count = 0
                    for i in range(0, len(lossgrads)):
                        if lossgrads[i] is None:
                            continue
                        if torch.max(torch.abs(lossgrads[i])).item() < self.lossgradtol:
                            count += 1
                    if count == len(lossgrads):
                        if rank == 0:
                            print(' '*12, 'lossgrads are small, stopping training')
                            self.saveModel()
                            io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                        self.running = False
                        logger.info('Minimization converged (lossgrads are small)')
                        if self.write_final_grads ==True:
                            final_lossgrads = open("lossgrads_final.txt", "w") 
                            final_lossgrads.write("Minimization converved: \n")
                            count =0
                            for i in range(0, len(lossgrads)):
                                if lossgrads[i] is None:
                                    continue
                                if torch.max(torch.abs(lossgrads[i])).item() < self.lossgradtol:
                                    count += 1
                                final_lossgrads.write("Final Max loss grads for parameter set "+str(i)+": "+str(torch.max(torch.abs(lossgrads[i])).item())+'\n')
                            final_lossgrads.write("Total Parameter Sets: "+str(len(lossgrads))+'\n')
                            final_lossgrads.write("Entire Loss Grad: \n")
                            final_lossgrads.write(str(lossgrads)+'\n')
                            #final_lossgrads.write('\n'+str(loss)+'\n')
 
                    self.nIter += 1

                    # add a condition to stop the training if the loss is too high
                    # if np.abs(cum_loss.item()) > 1000000 and self.nIter > 100:
                    #     self.running = False
                    #     if rank == 0:
                    #         print(' '*12, 'loss is too high, stopping training')

                    if self.nIter % self.logmodel_interval == 0:
                        # while saving the model, only make rank 0 do it
                        if rank == 0:
                            self.saveModel()
                            io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                    if energyRMSE < self.energyRMSEtol and forceRMSE < self.forceRMSEtol:
                        self.running = False
                        if rank == 0:
                            logger.info('Minimization converged (RMSEtol config)')
                            self.saveModel()
                            io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                    
                    if self.nIter >= maxEpochs and self.running == True: #we don't want the log file to have both Minimizatino Converged and Max Epoch Reached
                        self.running = False
                        if rank == 0:
                            logger.info('Max Epoch Reached')
                            print(' '*12,'Max Epoch Reached')
                            self.saveModel()
                            io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                        if self.write_final_grads ==True:
                            final_lossgrads = open("lossgrads_final.txt", "w") 
                            final_lossgrads.write("Max Epochs Reached: \n")
                            count =0
                            for i in range(0, len(lossgrads)):
                                if lossgrads[i] is None:
                                    continue
                                if torch.max(torch.abs(lossgrads[i])).item() < self.lossgradtol:
                                    count += 1
                                final_lossgrads.write("Final Max loss grads for parameter set "+str(i)+": "+str(torch.max(torch.abs(lossgrads[i])).item())+'\n')
                            final_lossgrads.write("Total Parameter Sets: "+str(len(lossgrads))+'\n')
                            final_lossgrads.write("Entire Loss Grad: \n")
                            final_lossgrads.write(str(lossgrads))
                            final_lossgrads.write('\n'+str(loss)+'\n')
                            
                    # make only rank 0 report the update to optimizer
                    if rank != 0:
                        optimizer.zero_grad()
                        #for p in self.model.parameters():
                        #    p.grad = None
                    
                    # print memory
                    # if rank == 0:
                    #     print_memory_info()

                    return float(cum_loss.item())

                if maxEpochs < 0: # Naman made this less than zero, if someone wants to restart the model and check if it restarted from the same spot
                    self.running = False

                while self.running:
                    optimizer.step(closure)
                    
                
                self.saveModel()
                io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")


        else:
            if self.optimizer == 'SGD':
                """
                    Uses torch SGD optimizer. Loss is not averaged over batches in order to save memory.
                """

                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learningRate,
                                            weight_decay = self.weight_decay)
            elif self.optimizer == 'ADAM':
                """
                    Uses torch ADAM optimizer. Loss is not averaged over batches in order to save memory.
                """
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate,
                                             weight_decay=self.weight_decay)

            elif self.optimizer == 'Rprop':
                """
                    Uses torch Rprop optimizer. Loss is not averaged over batches in order to save memory.
                """
                optimizer = torch.optim.Rprop(self.model.parameters(), lr=self.learningRate)

            #torch.cuda.set_device(device)
            if maxEpochs <= 0:
                self.running = False

            while self.running:

                if rank == 0:
                    optimizer.step()
                self.nIter += 1
                optimizer.zero_grad()
                loss, energylossRMSE, forcelossRMSE, \
                energyRMSE, forceRMSE, lossgrads, \
                charges_epoch, charge_percent_epoch = self.getLoss(batches, batches_Ele, device, logger)


                if reportTestRMSE and (self.nIter - 1) % self.test_loginterval == 0:
                    teRMSE, tfRMSE = self.testRMSE(testBatches, testBatches_Ele, device, logger, rank)
                else:
                    teRMSE, tfRMSE = False, False
                
                # update charge.log & percent_charge.log
                # Alan: averaged over batches
                if rank == 0:
                    for ele in charges_epoch.keys():
                        charges_epoch[ele] = torch.mean(charges_epoch[ele])
                    for type in charge_percent_epoch.keys():
                        charge_percent_epoch[type] = torch.mean(charge_percent_epoch[type])

                    self.update_logger(charge_logger, charges_epoch)
                    self.update_logger(charge_logger2, charge_percent_epoch)

                if rank == 0:
                    if teRMSE:
                        logger.info('%s', "{:12d} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}".format(self.nIter, loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE, teRMSE, tfRMSE))
                        print('%12d %12.6f %12.6f %12.6f %12.6f %12.6f' % (self.nIter, loss, energyRMSE, forceRMSE, teRMSE, tfRMSE))
                    else:
                        logger.info('%s', "{:12d} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}".format(self.nIter, loss, energylossRMSE, forcelossRMSE, energyRMSE, forceRMSE))
                        print('%12d %12.6f %12.6f %12.6f' % (self.nIter, loss, energyRMSE, forceRMSE))


                # Raise error if energy RMSE or/and force RMSE is nan
                if math.isnan(energyRMSE):
                    raise ValueError('energy RMSE is nan')
                   
                if math.isnan(forceRMSE):
                    raise ValueError('force RMSE is nan')
                count = 0
                for i in range(0, len(lossgrads)):
                    if lossgrads[i] is None:
                        continue
                    if torch.max(torch.abs(lossgrads[i])).item() < self.lossgradtol:
                        count += 1
                    #try: #if we want to ever try using the norm of the entire lossgrads as convergence
                    #   if torch.linalg.matrix_norm(lossgrads[i]).item() < self.lossgradtol:
                    #    count += 1
                    #except:
                    #   if torch.linalg.vector_norm(lossgrads[i]).item() < self.lossgradtol:
                    #    count += 1
                    #print ("Max loss grads: ",torch.max(torch.abs(lossgrads[i])).item())
                if count == len(lossgrads):
                    if rank == 0: # only rank 0 will do the writing
                        print(' '*12, 'lossgrads are small enough')
                        logger.info('Minimizaton converged')
                        self.saveModel()
                        io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                    if self.write_final_grads ==True:
                        final_lossgrads = open("lossgrads_final.txt", "w") 
                        final_lossgrads.write("Minimization converged: \n")
                        for i in range(0, len(lossgrads)):
                            if lossgrads[i] is None:
                                continue
                            if torch.max(torch.abs(lossgrads[i])).item() < self.lossgradtol:
                                count += 1
                            final_lossgrads.write("Final Max loss grads for parameter set "+str(i)+": "+str(torch.max(torch.abs(lossgrads[i])).item())+'\n')
                        final_lossgrads.write("Total Parameter Sets: "+str(len(lossgrads))+'\n')
                        final_lossgrads.write("Entire Loss Grad: \n")
                        final_lossgrads.write(str(lossgrads))
                    self.running = False

                if self.nIter % self.logmodel_interval == 0:
                    self.saveModel()
                    io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")

                if energyRMSE < self.energyRMSEtol and forceRMSE < self.forceRMSEtol:
                    if rank == 0: # only rank 0 will do the writing
                        logger.info('Minimizaton converged')
                        self.saveModel()
                        io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                    self.running = False

                if self.nIter >= maxEpochs and self.running ==True: #rare case if the optimization converged at last epoch
                    self.running = False
                    if rank == 0:
                        logger.info('Max Epoch Reached')
                        print(' '*12,'Max Epoch Reached')
                        self.saveModel()
                        io.saveFF(self.model, self.preprocessParas, filename="mlff.pyamff")
                    if self.write_final_grads ==True:
                        final_lossgrads = open("lossgrads_final.txt", "w") 
                        final_lossgrads.write("Max Epochs Reached: \n")
                        count =0
                        for i in range(0, len(lossgrads)):
                            if lossgrads[i] is None:
                                continue
                            if torch.max(torch.abs(lossgrads[i])).item() < self.lossgradtol:
                                count += 1
                            final_lossgrads.write("Final Max loss grads for parameter set "+str(i)+": "+str(torch.max(torch.abs(lossgrads[i])).item())+'\n')
                        final_lossgrads.write("Total Parameter Sets: "+str(len(lossgrads))+'\n')
                        final_lossgrads.write("Entire Loss Grad: \n")
                        final_lossgrads.write(str(lossgrads)+'\n')
                        #final_lossgrads.write('\n'+str(loss)+'\n')

        dist.destroy_process_group()
