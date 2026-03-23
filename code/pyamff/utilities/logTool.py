import sys, os
import platform, socket, re, uuid, subprocess
import logging

logo =\
"""
Long-range interaction qeq-v2
______       ___  ___  _______________ 
| ___ \     / _ \ |  \/  ||  ___|  ___|
| |_/ /   _/ /_\ \| .  . || |_  | |_   
|  __/ | | |  _  || |\/| ||  _| |  _|  
| |  | |_| | | | || |  | || |   | |    
\_|   \__, \_| |_/\_|  |_/\_|   \_|    
       __/ |                           
      |___/  
"""

def setLogger(name='pyamff', logfile='pyamff.log'):
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile)  
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    return logger

def writeSysInfo(logger):
    logger.info(logo)
    getSystemInfo(logger)

def getProcessorName():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "grep model /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode('utf-8').strip()
        #process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        #all_info, error = process.communicate()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line,1)
    return ""

def getSystemInfo(logger=None):
    if logger is None:
      sys.stderr('No logger is provided')
    try:
        logger.info('Platform:            %s', platform.system())
        logger.info('Platform-release:    %s', platform.release())
        logger.info('Platform-version:    %s', platform.version())
        logger.info('Architecture:        %s', platform.machine())
        logger.info('Hostname:            %s', socket.gethostname())
        logger.info('IP-address:          %s', socket.gethostbyname(socket.gethostname()))
        logger.info('MAC-address:         %s', ':'.join(re.findall('..', '%012x' % uuid.getnode())))
        logger.info('Processor:          %s', getProcessorName())
        #logger.info('Ram:                 %s', str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB")
    #except Exception as e:
    #    sys.stderr(e)
    except:
        pass

