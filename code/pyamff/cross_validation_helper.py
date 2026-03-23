
import numpy
from collections import OrderedDict
import torch
from pyamff.fingerprints.fingerprintsWrapper import atomCenteredFPs
from pyamff.utilities.preprocessor import normalizeParas
from pyamff.mlModels.pytorchNN import NeuralNetwork
from pyamff.fingerprints.fingerprints import Fingerprints
import os
import pickle

def train_test_split_new(images,num_test_img):
    #surprisingly painful function to write, test, and debug; surprisingly painful ...
    train = images.copy()
    test = []
    start=int(num_test_img/2) #floor round to integer
    if start ==0: # if we only need 1 testing image, get the first one
        test = [images[0]]
        train.pop(0)
    else: # otherwise get the first 50% from starting, and last 50% from ending
        test = images[-start:]+images[0:start]
        del (train[-start:],train[0:start])
        if num_test_img %2 ==1: # if odd number of images, get another one from the start
            test.append(images[start+1])
            train.remove(images[start+1])
        if (len(train)+len(test))!=len(images): #sanity check for/if somehow things break
            print ("FATAL TRAIN TEST SPLIT ERROR")
            return 1
    return train,test

### function to get an individual image's atomic U,F RMSEs
def image_mse(trueU, pred_U ,trueF,pred_F,num_atoms):
    U_mse =((pred_U-trueU)/num_atoms)**2
    F_mse = 0.0
    for vector in range(len(trueF)):
            F_err_compwise = 0
            for component in range(3): #x,y,z
                F_err_compwise +=(trueF[vector][component]-pred_F[vector][component])**2   # per vector per component force error
            atomic_image_F_err = F_err_compwise/(3*num_atoms) #atomwise sum - this should a number
            F_mse +=atomic_image_F_err  # get the Force MSE and add
    return U_mse,F_mse

def test_model(test_img_list,NN_model=None,Fingerprint_Class=None,training_fprange=None,num_fps=None,analyze=False):
    U_mse = 0
    F_mse = 0
    # for each image in the list of test strucutres
    # we firstly compute the fingerprint w.r.t fpRange of Training
    if analyze ==True: #maybe useful for Naman (or Me one day)
        image = 0
        analysis = open('analysis_test.txt','w')
        analysis.write("Image_Num  Energy_RMSE(atomic) Force_RMSE(atomic)\n")
    for test_image in test_img_list:
        #image+=1 # keep track of which image we are working with
        keylist = [0] # need it to sort fingerprints for pytorch
        chemsymbols = test_image.get_chemical_symbols() # get the elements in image
        fps, dfps = Fingerprint_Class.calcFPs(test_image, chemsymbols) #using Fortran compute the fingerprints and derivaties
        #print ("test fps: ",fps)
        acf = atomCenteredFPs() # initialize the python class to convert fortran fps to pytorch compatible format
        acf.sortFPs(chemsymbols, fps,num_fps,keylist=keylist, properties=None, fingerprintDerDB=dfps) # sort the fps and dfps
        acf.stackFPs([acf]) # stack in order for pytorch
        tfpRange, tmagnitudeScale, tinterceptScale = normalizeParas(training_fprange) #Normalize fingerprints ; note the fpRange is from training fps
        acf.normalizeFPs(tfpRange, tmagnitudeScale, tinterceptScale)# finish normalizing
        #End of Finperprint Calculation for the test structure

        #Now let's see how well we did
        #Get the true energy and force for the test structure
        true_U = test_image.get_potential_energy()
        trueF = test_image.get_forces()
        #print ("TEST: acf.allElement_fps: ",acf.allElement_fps)
        #Predict the Machine Learning Energy and Force
        pred_U, pred_F = NN_model.forward(acf.allElement_fps, acf.dgdx, acf, device=torch.device("cpu"))#using the python calculator, predict NN output
        # Forces are fine for Linear and NoScaler;  just dfps forward propagated through NN Model 
        # Energies have to be scaled w.r.t Scaler 
        ### Fix the python calc in asecalc.py as well ... this assumes self.adjust = False
        if NN_model.scalerType in ['LinearScaler']:
            pred_U = (pred_U + NN_model.scaler.intercept).data.numpy()[0]
            pred_F = pred_F.data.numpy() ## Only for Linear and NoScaler
        if NN_model.scalerType in ['STDScaler']:
            pred_U = (NN_model.slope*pred_U + NN_model.scaler.intercept*len(test_image)).data.numpy()[0]
            pred_F = (NN_model.scaler.f_std*pred_F).data.numpy() ## changed and tested
        if NN_model.scalerType in ['NoScaler']:
            pred_U = pred_U.data.numpy()[0]
            pred_F = pred_F.data.numpy() ## Only for Linear and NoScaler
        ###  Get the Image Energy and Force MSE 
        funcUMSE,funcFMSE = image_mse(true_U,pred_U,trueF,pred_F,len(test_image))
        U_mse +=funcUMSE
        F_mse +=funcFMSE
        if analyze ==True:
            analysis.write("   "+str(image)+"            "+str(funcUMSE**0.5)[0:8]+"           "+str(funcFMSE**0.5)[0:8]+"\n")
            image +=1
    ### Now get the loss (assuming force coefficient = 1.0
    #total_loss = (U_mse+F_mse)**0.5
    U_RMSE = (U_mse/len(test_img_list))**0.5
    F_RMSE = (F_mse/(len(test_img_list)))**0.5
    return  U_RMSE,F_RMSE
    #return total_loss, U_RMSE,F_RMSE

#Old test set split (just get first num images of traj file - code works; just not good way
#def test_split(traject_list,num_test_imgs):
#    return traject_list[0:num_test_imgs]

#Sample Usage of train_test_split_new
#test_ratio =0.1
#source_list = list(numpy.linspace(1,55,num=55))
#num_test =int(test_ratio*len(source_list))#this will floor round always
#print ('num_test: ',num_test)
#a,b = train_test_split_new(source_list,num_test)
#print ("Training Data: ",a)
#print ("Testing Data: ",b)
#print (len(a+b))

##### Mostly Working Code for k-fold cross validation (for now deprecated)

#def k_fold_split(traject_list,k_value):
#    num_imgs_per_fold = int(len(traject_list)/k_value)
    # looping for each num_imgs_per_fold sets
#    for i in range(0, len(traject_list), num_imgs_per_fold):
#        yield traject_list[i:i +num_imgs_per_fold]
     

#def fetchProp_validation(testImg,indexes):
#    images = OrderedDict()
#    properties = OrderedDict()
#    for i in range(len(testImg)):
#        index = indexes[i]
#        images[index] = testImg[i]
#        properties[index] = [numpy.array([testImg[i].get_potential_energy()]),numpy.array([testImg[i].get_forces()])]
#    return images,properties

#fname = os.path.join("./fingerprints1")

#def get_validation_acf(fname):
#    val_img_acf =[]
#    for validation_fps in os.listdir(fname):
#        if validation_fps.startswith('val'):#only the validation fps start with val
#            with open(os.path.join(fname,validation_fps), 'rb') as f:
                #print (validation_fps)
#                img_acf = pickle.load(f)
#                val_img_acf.append(img_acf)
#        if validation_fps.startswith('fpr'): #only fprange startwith fpr
#            with open(os.path.join(fname,validation_fps), 'rb') as f:
#                fpRange = pickle.load(f)
#    return val_img_acf,fpRange
#print ("fpval6: ",fpRange)
#print ("acf.energy: ",fpRange.energies)
#print ("acf.allElement_fps: ",fpRange.allElement_fps)
#print ("len(fpRange.allElement_fps): ",len(fpRange.allElement_fps))

#def num_atoms_in_img(fps_dict):
#    num_atoms = 0
#    for element in fpRange.allElement_fps:
#        num_atoms +=len(fpRange.allElement_fps[element])
#    return num_atoms

#print (num_atoms_in_img(fpRange))
#def cross_val_loss(NN_model,fpdir='fingerprints1'):
        # specificially not done with batches because we do not want backpropagation,
        # only forward calls

        #self.ddp_model = DDP(self.model, device_ids=[])
        #print ("device val loss: ",device)
        #device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#        print ("in validation resutls: ")
#        val_U_MSE = 0.0
#        val_F_MSE = 0.0
#        #if validationProp:
#        val_fps,training_fprange = get_validation_acf(fpdir)
#        for fps in val_fps:
            #trueU = fps.energies.data.numpy()[0]
            #trueF = fps.forces.data.numpy()
            #print ("True U: ",trueU)
            #trueF =
#            fpsRange,magnitudeScale,interceptScale = normalizeParas(training_fprange)
#            fps.normalizeFPs(fpsRange,magnitudeScale,interceptScale)
#            print ("Valid: ",fps.allElement_fps)
            #print ("self.model: ",self.model)
            #predEnergies, predForces  = self.ddp_model(fps.allElement_fps, fps.dgdx, fps, device2)
            #pred_U, pred_F = NN_model.forward(fps.allElement_fps, fps.dgdx, fps, torch.device("cpu"))
            ### Fix the python calc in asecalc.py and cross validation ...
            #if self.model.scalerType in ['LinearScaler', 'MinMaxScaler']:
            #    pred_U = (pred_U + seld.model.scaler.intercept).data.numpy()[0]
            #if self.model.scalerType in ['NoScaler']:
            #    pred_U = pred_U.data.numpy()[0]
            #pred_F = pred_F.data.numpy() ## Only for Linear and NoScaler
            ###  Get the Image Energy and Force MSE

            #print ('pred_U : ',pred_U)#
#        print ("val_fps : ",val_fps)
#        return val_U_MSE,val_F_MSE
