"""
Used for active learning
"""
import time
from pyamff.utilities.dataPartition import batchGenerator, Dataset, DataPartitioner
from pyamff.utilities.preprocessor import normalize, fetchProp, normalizeParas
from .training import Trainer
from pyamff.mlModels.pytorchNN import NeuralNetwork
from pyamff.neighborlist import NeighborLists
from pyamff.config import ConfigClass
from pyamff.fingerprints.behlerParrinello import represent_BP
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
#from pyamff.utilities.logTool import setlogger
import copy, os
from pyamff.fingerprints.fingerprints import Fingerprints
import torch
from pyamff.utilities.logTool import setLogger, writeSysInfo
import torch.distributed as dist

#logger = setLogger()
class activeLearn():

    def __init__(self, configfile, optim=False, params=None):
        # Read in parameters
        self.logger = setLogger()
        print(os.getcwd())
        writeSysInfo(self.logger)
        self.config = ConfigClass()
        self.config.initialize()
        self.params = params
        fp_paras = self.config.config['fp_paras'].fp_paras
        self.nFPs = {}
        for key in fp_paras.keys():
            self.nFPs[key] = len(fp_paras[key])
        self.trainingimages = []
        self.cycle = 0
        self.optim = optim
        self.active = False
        self.maxEpochs = self.config.config['epochs_max']
        self.losstol = self.config.config['force_coefficient'] * self.config.config['force_tol'] \
                     + self.config.config['energy_coefficient'] * self.config.config['energy_tol']
        self.model = NeuralNetwork(
                       hiddenlayers=self.config.config['hidden_layers'],
                       nFPs=self.nFPs,
                       forceTraining=self.config.config['force_training'],
                       # TODO: load pretrained params
                       params=self.params,
                       slope=None,
                       debug = self.config.config['nn_values']
                       )

        self.calc = Trainer(model=self.model,
                      optimizer=self.config.config['optimizer_type'],
                      # TODO: check if it can be reloaded
                      fpParas=self.config.config['fp_paras'],
                      energy_coefficient=self.config.config['energy_coefficient'],
                      force_coefficient=self.config.config['force_coefficient'],
                      lossConvergence=self.losstol,
                      energyRMSEtol=self.config.config['energy_tol'],
                      forceRMSEtol=self.config.config['force_tol'],
                      lossgradtol=self.config.config['loss_grad_tol'],
                      # TODO
                      intercept=None,
                      learningRate=self.config.config['learning_rate'],
                      model_logfile='pyamff.pt',
                      logmodel_interval=100,
                      debug=None,
                      weight_decay=self.config.config['weight_decay'],
                      fpRange=None)

        #self.fpDbs = OrderedDict()
        self.fpDbs = []
        self.uniq_elements=self.config.config['fp_paras'].uniq_elements
        #os.environ['MASTER_ADDR'] = '127.0.0.1'
        #os.environ['MASTER_PORT'] = '12355'
        os.environ['MASTER_ADDR'] = self.config.config['master_addr']
        os.environ['MASTER_PORT'] = self.config.config['master_port']
        dist.init_process_group('gloo', rank=0, world_size=1)


    def train(self, images, debug=False, params=None):
        #logg = self.logger
        #self.trainingimages.append(image)

        if self.config.config['use_cohesive_energy']:
            coeh = self.config.config['fp_paras'].refEs
            refEs = OrderedDict(zip(self.uniq_elements, coeh))
        else:
            refEs = None

        trainingimages, properties, slope, intercept = fetchProp(images, refEs=refEs, forceTraining=self.config.config['force_training'], activeLearning=True)
        nimages = len(trainingimages)
        # Get neighborlists
        #TODO Set up active learning with python only
        """
        nl = NeighborLists(cutoff=6.0)
        nl.calculate(trainingimages, fortran=True)

        # Calculate fingerpints of the new image: fpRange={} case

        acfs, fpRange = represent_BP(nl, self.nFPs, trainingimages, properties, G_paras=self.config.config['fp_paras'].fp_paras, fortran=True, normalize=False)
        for key in acfs.keys():
            self.fpDbs.append(acfs[key])
        print(' # of images', len(self.fpDbs))
        acfs = atomCenteredFPs()
        acfs.stackFPs(copy.copy(self.fpDbs))
        #if self.cycle == 0:
        #  #self.acfs = copy.copy(acfs[self.numberImages])
        #  self.acfs.stackFPs(list(acfs.values()), new=True, activelearning=True)
        #else:
        #  self.acfs.stackFPs(list(acfs.values()), new=False, activelearning=True)
        acfs.findFPrange()
        #self.acfs.setOriginal()
        slope, intercept = acfs.scaleEnergies()

        fpRange, magnitudeScale, interceptScale = normalizeParas(acfs.fpRange)
        """
        useexisting = self.config.config['fp_use_existing']
        if useexisting:
            self.logger.info('Using pre-calculated fingerprints from %s', cwd+'/'+self.config.config['fp_dir'])
            #print('%.2fs: Using pre-calculated fingerprints'%(time.time()-st))
            fpDir = self.config.config['fp_dir']
            #nfpFiles = len(glob.glob(os.path.join(os.getcwd(), 'fingerprints/*')))
            nfpFiles = len(glob.glob(os.path.join(fpDir, '*')))
            if nfpFiles != nimages:
                print(" Not enough fingerprint files: %d expected but only %d provided" % (nimages, nfpFiles), sys.stderr)
                sys.exit(2)
        else:
            fpDir = None
        if self.optim and self.cycle != 0:
            self.active=True

        fpcalc = Fingerprints(uniq_elements=self.uniq_elements, filename=self.config.config['fp_parameter_file'], nfps=self.nFPs,active=self.active)
        fpRange, magnitudeScale, interceptScale = fpcalc.loop_images(self.nFPs, self.config.config['fp_batch_num'], trainingimages, properties,
                                   normalize=True, logger=self.logger, fpDir=fpDir, useexisting=useexisting)

        setattr(self.model, 'slope', slope)
        setattr(self.calc, 'intercept', intercept)
        setattr(self.calc, 'fpRange', fpRange)
        setattr(self.calc, 'nimages', len(trainingimages))
        #acfs.normalizeFPs(fpRange, magnitudeScale, interceptScale)i
        srcData = list(trainingimages.keys())
        process_number = 1
        tbatches = 1
        st = time.time()
        partitions = DataPartitioner(srcData=srcData,
                                     fpRange=fpRange,
                                     magnitudeScale=magnitudeScale,
                                     interceptScale=interceptScale,
                                     process_numb=process_number,
                                     fpDir=fpDir,
                                     batch_numb=tbatches,
                                     seed=1234,
                                     st=st)

        if debug:
            self.model.setParams(params)
        #partition = Dataset({0:self.acfs}, [0])
        useCuda = False

        if self.config.config['device_type'] == 'GPU' and torch.cuda.is_available():
            useCuda = True

        device = torch.device("cuda:0" if useCuda else "cpu")
        part = partitions.use(0)
        self.calc.parallelFit(0, 1, 1,
                              part, 1, self.maxEpochs, device, logger=self.logger)
        #self.acfs.fetchOriginal()
        self.cycle += 1

        return self.model, self.calc.preprocessParas
