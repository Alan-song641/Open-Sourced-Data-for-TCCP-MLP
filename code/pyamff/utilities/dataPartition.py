import numpy as np
import torch
import copy, math
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
from .fileIO import loadData
from collections import OrderedDict
import random, time
import os
import pickle
import bz2file as bz2
import glob

def partitionData(indexes, nBatches, seed=None):
    """
    Partition fp data to 'nBatches' batches
    """
    batches = {}
    sizes = [1.0 / nBatches for _ in range(nBatches)]
    data_len = len(indexes)
    if seed:
        random.seed(seed)
        random.shuffle(indexes)
    batchID = 0
    for frac in sizes:
        part_len = int(frac*data_len)
        batches[batchID] = indexes[0:part_len]
        indexes = indexes[part_len:]
        batchID+=1
    if len(indexes)>0:
        #print("Warning batch size of one partition is smaller than others: %d vs %d"%(len(indexes), part_len), flush=True) 
        batchID = 0
        for index in indexes:
            batches[batchID].append(index)
            batchID+=1
    """
    for frac in sizes:
        part_len = int(frac * data_len)
        if self.inmemory:
            batches[batchID]  =  self.preprocess(indexes[0:part_len], batchID)
        else:
            self.preprocess(indexes[0:part_len], batchID)
            batches[batchID] = batchID
        print('indexes:', batchID, indexes[0:part_len])
        indexes = indexes[part_len:]
        batchID+=1
        if frac == sizes[-1] and len(indexes) < part_len and len(indexes)>0:
            print("Warning batch size of one partition is smaller than others: %d vs %d"%(len(indexes), part_len), flush=True) 
    """
    return batches

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    internal_cache = {}
    def __init__(self, data, srcDir, indices):
          'Initialization'
          self.data = data
          self.srcDir = srcDir
          self.indices = indices
          self.internal_cache = {}

    def __len__(self):
          'Denotes the total number of samples'
          return len(self.indices)

    def __getitem__(self, index):
          'Generates one sample of data'
          # Select sample
          #print('fetch', self.indices[index])
          if index in self.internal_cache:
              return self.internal_cache[index]

          fname = os.path.join(self.srcDir,
                               'batches_{}.pckl'.format(self.indices[index]))
          with bz2.BZ2File(fname, 'rb') as f1:
              X = pickle.load(f1)
          #X = self.data[self.indices[index]]
          self.internal_cache[index] = X
          return X


class Dataset_Eele(torch.utils.data.Dataset):
    'Characterizes a dataset (batches_efps_{}.pckl) for PyTorch'

    def __init__(self, data, srcDir, indices):
        'Initialization'
        self.data = data
        self.srcDir = srcDir
        self.indices = indices

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #print('fetch', self.indices[index])
        fname = os.path.join(
            self.srcDir, 'batches_efps_{}.pckl'.format(self.indices[index]))
        with bz2.BZ2File(fname, 'rb') as f1:
            X = pickle.load(f1)
        #X = self.data[self.indices[index]]
        return X

class DataPartitioner(object):
    """
    Used to split data based on number of process
    srcData: a list of keys that used to store fp data
    """
    def __init__(self, srcData, fpRange, magnitudeScale,
                 interceptScale, nProc, fpDir=None, nBatches=None, useExisting=False, seed=1234, test=False, st=None):
        #print('%.2fs: Entering dataPartition'%(time.time()-st))
        # d = device # i dont want to send device variable everywhere hence made it d
        # if isinstance(srcData, str):
        #     self.inmemory = True
        #     data = load_data(filename=srcData, rb='rb')
        #     self.fpRange = data['fpRange']
        #     self.fpData = data['fpData']
        if isinstance(srcData, list):
            self.inmemory = False
            self.fpData = srcData 
            self.fpRange = fpRange
            self.magnitudeScale = magnitudeScale
            self.interceptScale = interceptScale
        self.partitions = []
        if fpDir is None:
            if test: # for train_testFF
                self.srcDir = os.getcwd() + '/test_fingerprints'
            else:
                self.srcDir = os.getcwd() + '/fingerprints'
        else:
            self.srcDir = fpDir
        if test: # for train_testFF
            self.batchDir = os.getcwd() + '/test_batches'
        else:
            self.batchDir = os.getcwd() +'/batches'
        if not os.path.exists(self.batchDir):
            os.mkdir(self.batchDir)
        #print ("BATCH NUMBER: ",batch_numb)
        self.nBatches = nBatches
        if nBatches:
            if self.inmemory:
                indexes = list(self.fpData.keys())
            else:
                indexes = copy.copy(self.fpData)
            batches = partitionData(indexes, nBatches, seed)
            for key in batches.keys():
                #print('indexes:', key, batches[key])
                # if self.inmemory:
                #     batches[batchID] = self.preprocess(batches[key], d, key)
                # else:
                self.preprocess(batches[key], useExisting, key)
                batches[key] = key
            self.fpData = batches
        self.partitionBatches(nProc, seed)
        print('%8.2fs: Data partition done' % (time.time()-st))


    def preprocess(self, dataIDs, useExisting, batchID):
        if useExisting:
            print('Trying to load training batches')
            # fpDir = 'batches'
            #nfpFiles = len(glob.glob(os.path.join(os.getcwd(), 'fingerprints/*')))
            nfpFiles = len(glob.glob(os.path.join(self.batchDir, '*'))) 
            if nfpFiles != self.nBatches*2:
                #print("Inconsistant number of fingerprint files: %d expected but %d found" % (nImages, nfpFiles), sys.stderr)
                print("Inconsistant number of fingerprint files: %d expected but %d found" % (self.nBatches*2, nfpFiles))
                print(' '*12,'Batches were not found')
                #sys.exit(2)
                useExisting = False
            else:
                # logger.info('Using pre-calculated fingerprints from %s', cwd+'/'+config.config['fp_dir'])
                print('Loading precalculated training batches')
                return
            # print('[INFO]   useExisting = True, skipping batching')
            
        
        acfs = []
        # Alan: ewald_FPs
        efps = []

        # for any mode, if batches exist, then use it
        fname = os.path.join(self.batchDir, 'batches_{}.pckl'.format(batchID))
        cname = os.path.join(self.batchDir, 'batches_efps_{}.pckl'.format(batchID))
        if os.path.exists(fname) and os.path.exists(cname):
            with bz2.BZ2File(fname, 'rb') as f1:
                acfs = pickle.load(f1)
            with bz2.BZ2File(cname, 'rb') as f2:
                efps = pickle.load(f2)
                return acfs, efps

        for dataID in dataIDs:
            fname = os.path.join(self.srcDir, 'fps_{}.pckl'.format(dataID))
            with bz2.BZ2File(fname, 'rb') as f1:
                acf = pickle.load(f1)
                #acf.normalizeFPsList(self.fpRange, self.magnitudeScale, self.interceptScale)
                acf.normalizeFPs(self.fpRange, self.magnitudeScale, self.interceptScale)
                acfs.append(acf)

            cname = os.path.join(self.srcDir, 'efps_{}.pckl'.format(dataID))
            with bz2.BZ2File(cname, 'rb') as c1:
                efp = pickle.load(c1)
                #acf.normalizeFPsList(self.fprange, self.magnitudeScale, self.interceptScale)
                efps.append(efp)

        batch_acfs = batchGenerator(acfs)
        # if device:
        #     batch_acfs.toTensor(device=device) # we use this only when GPU is used for training, otherwise the if statement is skipped
        fname = os.path.join(self.batchDir, 'batches_{}.pckl'.format(batchID))
        with bz2.BZ2File(fname, 'wb') as f1:
            pickle.dump(batch_acfs, f1)
        
        # Alan: add acfs AND efps into batches.pckl
        batch_efps = batchGenerator_ele(efps)
        cname = os.path.join(self.batchDir,
                             'batches_efps_{}.pckl'.format(batchID))
        with bz2.BZ2File(cname, 'wb') as f1:
            pickle.dump(batch_efps, f1)

        return batch_acfs, batch_efps

    # Partition data based on number of process
    def partitionBatches(self, nProc, seed=1234):
       """
       Partition batches to 'nProc' set
       """
       sizes = [1.0 / nProc for _ in range(nProc)]
       random.seed(seed)
       indexes = list(self.fpData.keys())
       data_len = len(indexes)
       random.shuffle(indexes)
       for frac in sizes:
           part_len = int(frac * data_len)
           self.partitions.append(indexes[0:part_len])
           indexes = indexes[part_len:]

    def use(self, rank):
        return Dataset(self.fpData, self.batchDir, self.partitions[rank])
    
    def use_Ele(self, rank):
        return Dataset_Eele(self.fpData, self.batchDir, self.partitions[rank])


# collate_fn: imagesFPs: a list of atomsCenteredFps
def batchGenerator(acfs):
    st = time.time()
    if len(acfs) == 1: 
        #print (acfs)
        #print (acfs[0])
        batch = atomCenteredFPs()
        batch.stackFPs(acfs)
        return batch
    batch = atomCenteredFPs()
    #print("Batch: ",batch)
    #print ('ACFS: ',acfs)
    batch.stackFPs(acfs)
    #print("After Batch: ",batch)
    #batch.stackFPsList(acfs)
    #print(' BatchingTIMEUSED:', time.time()-st)
    return batch

def batchGenerator_ele(efps):
    '''
    
    efps:OrderDict{0:efp, 1:efp ...}
    '''

    # batch = Ewald_FPs()

    # batch.stackFPs(efps)

    return efps

