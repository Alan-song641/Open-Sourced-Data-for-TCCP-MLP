import pickle
import os
import torch
import numpy as np

def saveData(data, filename='fps.pckl', wb='wb', debug =False):
    if hasattr(filename, 'write'):
        fileout=filename
    else:
        fileout = open(filename, wb)

    pickle.dump(data,fileout)
    fileout.close()
    if debug == True:
        f = open('debug.txt', 'w')
        f.write(str(data))
        f.close()

def loadData(filename='fps.pckl', rb = 'rb'):
    try:
        filein=filename
        filein = open(filename, rb)
        data = pickle.load(filein)
        filein.close()
        return data
    except:
        print('error loading FP data')

def saveFF(model, preprocessParas, filename="mlff.pyamff"):
    f = open(filename, 'w')
    fp_type = preprocessParas['fingerprints'].type    #config.config['fp_paras']
    fps = preprocessParas['fingerprints'].fp_paras    #config.config['fp_paras']
    fpRange = preprocessParas['fpRange']
    f.write('#Fingerprint type\n')
    f.write('{:s}\n'.format(fp_type))
    G1s = []
    G2s = []
    elements = fps.keys()
    # Rmins set 0.01 for now
    nelements=len(elements)
    rmins=np.zeros([nelements,nelements])
    rmins.fill(0.01)
    f.write('#Rmins\n')
    for element in elements:
        f.write('{:>4s}'.format(element))
    f.write('\n')
    for j in range(nelements):
        for k in range(nelements):
            f.write('{:>8.2f}'.format(rmins[j][k]))
        f.write('\n')

    for key in elements:
        if key not in fpRange.keys():
            continue
        for i in range(len(fps[key])):
          fp = fps[key][i]
          setattr(fp, 'fmin', fpRange[key][0][i])
          setattr(fp, 'fmax', fpRange[key][1][i])
          if fps[key][i].subtype == 'G1':
              G1s.append(fp)
          if fps[key][i].subtype == 'G2':
              G2s.append(fp)
    if len(G1s) > 0:
       f.write('#{:>8s} {:>8s}\n'.format('type', 'number'))
       f.write('{:>8s} {:>8d}\n'.format('G1', len(G1s)))
       f.write('#{:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>12s} {:>12s}\n'.format(
               'center', 'neighbor', 'eta', 'Rs', 'rcut', 'fpmin', 'fpmax'))
       for g in G1s:
           f.write('{:>8s} {:>8s} {:>8.3f} {:>8.3f} {:>8.3f} {:>.10e}  {:>.10e}\n'.format(
                   g.center, g.neighbor, g.eta, g.Rs, g.rcut,  g.fmin, g.fmax))

    if len(G2s) >0 :
       f.write('#{:>8s} {:>8s}\n'.format('type', 'number'))
       f.write('{:>8s} {:>8d}\n'.format('G2', len(G2s)))
       f.write('#{:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>12s} {:>12s}\n'.format(
               'center', 'neighbor1','neighbor2', 'eta', 'zeta', 'lambda', 'thetas', 'rcut', 'fpmin', 'fpmax'))
       for g in G2s:
           f.write('{:>8s} {:>8s} {:>8s} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>.10e} {:>.10e}\n'.format(
                   g.center, g.neighbor1, g.neighbor2, g.eta, g.zeta, g.lambda_, g.thetas, g.rcut, g.fmin, g.fmax))

    # Model
    f.write('#MachineLearning model type\n')
    f.write('atomic-NN\n')
    f.write('#Activation function type\n')
    f.write('{:s}\n'.format(model.activation))
    f.write('#Model Structure\n')
    f.write('{:d}\n'.format(len(model.hiddenlayers)))
    f.write('{:s}\n'.format(' '.join([str(i) for i in model.hiddenlayers])))
    f.write('#Model Parameters\n')

    modelParas = model.state_dict().items()
    head = {}
    for element in elements:
        head[element] = False
    for name, value in modelParas:
#        print(name, value)
        if name in ['slope', 'intercept']:
            #continue
            if model.adjust: # Store adjusted slope and intercept
                if name == 'slope':
                    adjusted_slope=value
                if name == 'intercept':
                    adjusted_intercept=value
            continue
        fields = name.split('.')
        element = fields[1]
        parasName = fields[2]+ ' ' + fields[3]
        #print('wb:',name, value)
        if not head[element]:
            # Write a header
            f.write(' {:s}\n'.format(element))
            f.write("{:s} #{:s}\n".format(formatJoint(torch.flatten(value).cpu().numpy()), parasName))
            head[element]=True
        else:
            f.write("{:s} #{:s}\n".format(formatJoint(torch.flatten(value).cpu().numpy()), parasName))
    if model.scalerType in ['NoScaler']:
        f.write('#Energy Scaling Parameters\n')
        f.write('{:s}\n'.format(model.scalerType))
        f.write('{:<16.8f} {:<16.8f} #slope intercept\n'.format(preprocessParas['slope'], preprocessParas['intercept']))
    if model.scalerType in ['LinearScaler', 'MinMaxScaler', 'STDScaler']:
        f.write('#Energy Scaling Parameters\n')
        f.write('{:s}\n'.format(model.scalerType))
        if model.adjust:
            f.write('{:<16.8f} {:<16.8f} #slope intercept\n'.format(adjusted_slope, adjusted_intercept))
        else:
            f.write('{:<16.8f} {:<16.8f} #slope intercept\n'.format(preprocessParas['slope'], preprocessParas['intercept']))
    if model.scalerType in ['LogScaler']:
        f.write('#Min-Max Normalization Parameters\n')
        f.write('{:<16.8f} {:<16.8f} #energy min max\n'.format(model.eMinMax[0], model.eMinMax[1]))
        try:
            f.write('{:<16.8f} {:<16.8f} #force min max\n'.format(model.fMinMax[0], model.fMinMax[1]))
        except:
            pass
    f.close()

def formatJoint(array):
    jointed = ''
    for element in array:
        jointed+=" {:.8e}".format(element)
    return jointed

