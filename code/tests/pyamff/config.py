import configparser
import yaml
import os, sys
import copy
import logging

logger = logging.getLogger('pyamff')

class BP_paras:
    def __init__(self, subtype, n_element, n_variable):
        self.type = 'Behler-Parrinello'
        self.subtype = subtype
        self.n_element = n_element
        self.n_variable = n_variable
        #self.elementtable = ['H','Pd','Au']
        self.fpmin = None
        self.fpmax = None

    def setFPrange(self, fmin, fmax):
        self.fmin = fmin
        self.fmax = fmax
        #print(self.fmin)

    def checkFormat(self, fields):
        config_err = False
        if len(fields)!=self.n_variable:
            config_err = True
            sys.stderr.write('%d values are required, but only %d given\n'%(self.n_variable, len(fields)))
            return config_err
        else:
            #for i in range(1,self.n_element+1):
            #    if fields[i] not in self.elementtable:
            #        config_err = True
            #        sys.stderr.write('Element type s% does not exist in the periodic table\n'%(fields[i]))
            for i in range(self.n_element+1, self.n_variable):
                try:
                    x = float(fields[i])
                except:
                    config_err = True
                    sys.stderr.write('Failed to convert %s to float.\n'%(fields[i]))
                    pass
        return config_err

class G1(BP_paras):
    """
    Class to define the format of G1 fingerprint parameters

    """
    def __init__(self, neighbor=None, eta=None, Rs=None, center=None, rcut=None):
        super(G1, self).__init__(subtype='G1', n_element=2, n_variable=6)
        self.Rs = Rs
        self.eta = eta
        self.neighbor = neighbor
        self.center = center
        self.rcut = rcut

    def format(self):
        sys.stderr.write('G1 format: G1 <centerElement> <neighborElement> <eta> <Rs>\n')

class G2(BP_paras):
    """
    Class to define the format of G1 fingerprint parameters

    """
    def __init__(self, neighbor1=None, neighbor2=None, eta=None, zeta=None, lambda_=None, thetas=None, center=None, rcut=None):
        super(G2, self).__init__(subtype='G2', n_element=3, n_variable=9)
        self.eta = eta
        self.zeta = zeta
        self.lambda_ = lambda_
        self.thetas = thetas
        self.neighbor1 = neighbor1
        self.neighbor2 = neighbor2
        self.center = center
        self.rcut = rcut

    def format(self):
        sys.stderr.write('G2 format: G2 <centerElement> <neighborElement1> <enighborElement2> \
                          <eta> <zeta> <lambda> <thetas>\n')

class FingerprintsParas:
    def __init__(self, type='BP'):
        self.type = type
        self.fp_paras = {}
        self.uniq_elements = []
        self.refEs = []
        if self.type == 'BP':
            self.nG1s = 0
            self.nG2s = 0
        # logger.info('  Fingerprints used:')
        # logger.info('    Type: %s', type)
        # print(' '*12,'Fingerprints used:')
        # print(' '*12,'Type: ', type)

    def checkFormat(self,lines):
        for i in range(7):
            line = lines[i]
            if i in [0, 2, 5]:
                if line.startswith('#'):
                    continue
                else:
                    config_err = True
                    sys.stderr.write("Line %d should start with Symbol '#'\n"%(i+1))
                    return config_err
            if i == 1:
                if line.split()[0] != self.type:
                    config_err = True
                    sys.stderr.write("Fingerprint type does NOT match with that defined in 'config.ini' (Line 2)\n")
            if i == 3:
                try:
                    self.uniq_elements = [field for field in line.split()]
                except:
                    config_err = True
                    sys.stderr.write('Line %d should be unique element types included in the atomic structure\n'%(i+1))
            if i == 4:
                try:
                    self.refEs = [float(field) for field in line.split()]
                except:
                    config_err = True
                    sys.stderr.write('Line %d should be float types used to calculate cohensive energy\n'%(i+1))

            if i == 6:
                try:
                    fields = line.split()
                    self.nG1s = int(fields[0])
                    self.nG2s = int(fields[1])
                except:
                    config_err = True
                    sys.stderr.write('Line %d should be number of G1s followed by number of G2s\n'%(i+1))
        nG1s = 0
        nG2s = 0
        for line in lines[7:]:
            if line.startswith('#'):
                continue
            fields = line.split()
            if fields[0] == 'G1':
                g = G1()
                nG1s += 1
            elif fields[0] == 'G2':
                g = G2()
                nG2s += 1
            else:
                config_err = True
                sys.stderr.write('unknown fingerprint type: %s\n'%(fields[0]))
            config_err = g.checkFormat(fields)
        if nG1s != self.nG1s:
            config_err = True
            sys.stderr.write('Number of G1 fingerprints does NOT match: nG1s = %d but %d given\n'%(self.nG1s, nG1s))
        if nG2s != self.nG2s:
            config_err = True
            sys.stderr.write('Number of G2 fingerprints does NOT match: nG2s = %d but %d given\n'%(self.nG2s, nG2s))
        return config_err

    def fp_summary(self):
        # Used to print summary information of fps
        fps = {}
        for key in self.fp_paras.keys():
            fps[key] = {}
            for fp in self.fp_paras[key]:
                if fp.subtype == 'G1':
                    try:
                        fps[key]['G1'] += 1
                    except:   
                        fps[key]['G1'] = 1
                if fp.subtype == 'G2':
                    try:
                        fps[key]['G2'] += 1
                    except:   
                        fps[key]['G2'] = 1
        fpsummary = {}
        for key in fps.keys():
            temp = "{:2s}:".format(key)
            for gtype in fps[key].keys():
                temp += "{:>4s}".format(str(fps[key][gtype]))
                temp += "{:>3s}s".format(gtype)
            fpsummary[key]=temp
            # logger.info('    %s',fpsummary[key])
            # print(' '*12,fpsummary[key])
        return fpsummary

    def set(self, fp_file):
        fp = open(fp_file, 'r')
        lines = fp.readlines()
        fp.close()

        config_err = self.checkFormat(lines)
        if config_err:
            return config_err

        for line in lines[7:]:
            if line.startswith('#'):
                continue
            fields = line.split()
            if fields[0] == 'G1':
                try:
                    self.fp_paras[fields[1]].append(G1(neighbor=fields[2], eta=float(fields[3]), Rs=float(fields[4]), rcut=float(fields[5]), center=fields[1]))
                except:
                    self.fp_paras[fields[1]]=[G1(neighbor=fields[2], eta=float(fields[3]), Rs=float(fields[4]), rcut=float(fields[5]), center=fields[1])]
            if fields[0] == 'G2':
                try:
                    self.fp_paras[fields[1]].append( G2(neighbor1=fields[2], neighbor2=fields[3],
                                                        eta=float(fields[4]), zeta=float(fields[5]),
                                                        lambda_=float(fields[6]), thetas=float(fields[7]), rcut=float(fields[8]), center=fields[1])
                                                   )
                except:
                    self.fp_paras[fields[1]] = [ G2(neighbor1=fields[2], neighbor2=fields[3],
                                                    eta=float(fields[4]), zeta=float(fields[5]),
                                                    lambda_=float(fields[6]), thetas=float(fields[7]), rcut=float(fields[8]), center=fields[1])
                                               ]

class ConfigClass:
    """
    Parameters are stored in a dictionary:
      {'run_type': 'trainFF', 
       'trajectory_file': 'train.traj',
       'energy_training': True,
       'force_training': True,
       'fp_type': 'BP',
       'parameter_file': 'fpParas.dat',
       'model_type': 'neural_network',
       'loss_type': 'rmse',
       'energy_coefficient': 1.0,
       'force_coefficient': 0.02,
       'optimizer_type': 'LBFGS',
       'loss_convergence': 0.01,
       'fp_paras': <__main__.FingerprintsParas object at 0x7f1c79867fd0>}

        To fetch the particular parameter, use 'config.config[<nameOFparameter>]'
           i.e., fetch run_type:
            1. Define ConfigClass object and read 'config.yaml' (default values)
               config = ConfigClass()
            2. read 'config.ini' (user defined) 
               config.initialize()
            3. Fetch the parameter
               config.config['run_type']
        Fingerprint parameters are sorted as an object FingerprintsParas. It can be fetched with
           1. Fetch the FingerprintsParas object
              fp = config.config['fp_paras']
           2. Fetch the dictionary containing the fingerprint object
              fp.fp_paras
              it will give:
                {'H': [G1, G1, ..., G2, ...],
                 'Pd': [G1, G1, ..., G2, ...]
                }
              where G1 and G2 are objects containing the necessary parameters.
               i.e., neighbor, eta, Rs for G1
                     neighbor1, neighbor2, eta, zeta, lambda, thetas for G2

    """
    def __init__(self):
        self.init_done = False
        self.cwd = os.getcwd() + '/'
        self.config = {}

        self.parser = configparser.ConfigParser()

        yaml_file = open(os.path.join(os.path.dirname(__file__), 'config.yaml'))
        self.config_defaults = yaml.load(yaml_file, Loader=yaml.BaseLoader)
        yaml_file.close()

        # Pass default settings to parser
        for sectionName in self.config_defaults:
            self.parser.add_section(sectionName)
            for key in self.config_defaults[sectionName]['options']:
                kattr = self.config_defaults[sectionName]['options'][key]
                self.parser.set(sectionName, key.lower(), kattr['default'])
                #if 'values' in kattr:
                #    for value in kattr['values']:
                #        ck.values.append(value)
            #self.section_options[sectionName] = copy.deepcopy(self.parse.options(sectionName))
        #print(self.config_defaults)

    # Check legality and set values
    def setValues(self, data_type, section, vname, values=None):
        config_error = False
        if data_type == 'integerlist':
            try:
                self.config[vname] = self.parser.get(section, vname)
                self.config[vname] = tuple([int(field) for field in self.config[vname].split()])
            except:
                config_error = True
                sys.stderr.write('Option "%s" in section "%s" should be an integer\n' % (vname,section))
                pass
        elif data_type == 'integer':
            try:
                self.config[vname] = self.parser.getint(section, vname)
            except:
                config_error = True
                sys.stderr.write('Option "%s" in section "%s" should be a integer\n' % (vname,section))
        elif data_type == 'float':
            try:
                self.config[vname] = self.parser.getfloat(section, vname)
            except:
                config_error = True
                sys.stderr.write('Option "%s" in section "%s" should be a float\n' % (vname,section))
        elif data_type == 'boolean':
            try:
                self.config[vname] = self.parser.getboolean(section, vname)
            except:
                config_error = True
                sys.stderr.write('Option "%s" in section "%s" should be a boolean\n' % (vname,section))
                pass
        elif data_type == 'string':
            if values is not None:
                pvalue = self.parser.get(section, vname)
                if pvalue not in values:
                    config_error = True
                    sys.stderr.write('Option "%s" should be one of: %s\n' % (vname, ", ".join(values)))
                else:
                    self.config[vname] = pvalue
            else:
                self.config[vname] = self.parser.get(section, vname)
        return config_error


    def initialize(self, config_file="config.ini"):
        if os.path.isfile(config_file):
            self.parser.read(config_file)
            self.config_path = os.path.abspath(config_file)
        else:
            # TODO: config_path is not an attribute yet
            print("Specified configuration file %s does not exist" % ''.join([self.cwd,config_file]), sys.stderr)
            sys.exit(2)

        # Check that all sections in config.ini are in the configparser
        config_error = False
        psections = self.parser.sections()
        fsections = list(self.config_defaults.keys())
        main_diff = list(set(psections) - set(fsections))
        if len(main_diff) > 0:
            config_error = True
            sys.stderr.write('unknown sections "%s"\n' % ", ".join(main_diff))

        # Check that all options in config.ini are in the configparser
        for section in psections:
            b = self.parser.options(section)
            section_diff = list(set(b) - set(self.config_defaults[section]['options'].keys()))
            if len(section_diff) > 0:
                config_error = True
                sys.stderr.write('unknown option "%s" in section "%s"\n' % (", ".join(section_diff), section))

        # Check type of input settings and assignments
        for section in psections:
            defaults = self.config_defaults[section]['options']
            for op in self.parser.options(section):
                if 'values' in defaults[op]:
                    config_err = self.setValues(defaults[op]['kind'], section, op, values=defaults[op]['values'])
                else:
                    config_err = self.setValues(defaults[op]['kind'], section, op)
        # logger.info('Reading fingerprint parameters from %s', self.cwd+self.config['fp_parameter_file'])
        # Check if fpParas was given or we need to calculate GR
        if self.config['gr_calc'] == True:
            from tools.python_gr.amain import GRcalc
            logger.info('we need to calculate GR to make fpParas.dat')
            if os.path.isfile(self.config['trajectory_file']):
                GRcalc(self.config['trajectory_file'], self.config['gr_cutoff'], self.config['gr_process_num'], logger)

        self.config['fp_paras'] = FingerprintsParas()
        config_err = self.config['fp_paras'].set(fp_file=os.path.dirname(self.config_path) +'/'+ self.config['fp_parameter_file'])
        self.config['fp_paras'].fp_summary()
        if config_err:
            sys.exit(2)

if __name__ == "__main__":
    config = ConfigClass()
    config.initialize()
    print("Parameters are stored in a dictionary:")
    print("  ",config.config)
    print("    To fetch the particular parameter, use 'config.config[<nameOFparameter>]'")
    print("      i.e., fetch run_type with config.config['run_type']")
    print("          run_type:", config.config['run_type'])
    print("  Fetch fingerprint parameters")
    print(config.config['fp_paras'].fp_paras)
