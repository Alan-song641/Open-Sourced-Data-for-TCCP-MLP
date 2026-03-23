import torch.nn as nn
import torch

class LossFunction(nn.Module):
    def __init__(self, loss_type='SE', 
                       cohe = False,
                       energyCoefficient=1.00,
                       forceCoefficient=0.05, 
                       device=torch.device("cpu")):
        super().__init__()
        self.loss_type = loss_type
        self.cohe = cohe
        self.energyCoefficient = energyCoefficient
        self.forceCoefficient = forceCoefficient
        self.energyloss = None
        self.forceloss  = None
        self.energyRMSE = None
        self.forceRMSE = None
        self.device = device

    """
    inputs: [energies, forces]
          energies: tensor([E_1, E_2, ..., E_N])
          forces: tensor([f_1, f_2, ..., f_3N])
          N is the total number of atoms in the tranining set
    target: [energy_reference, forces_reference]
    """
    #def set_inputs(self, input1, input2, target1, target2, natomsEnergy, natomsForce):

def calc_mse_noForce(input1, target1, natomsEnergy):
    energyRMSE = torch.sum(
        torch.pow(torch.div(torch.sub(input1, target1), natomsEnergy), 2.))
    # forceRMSE = torch.sum(
    #     torch.div(torch.sum(torch.pow(torch.sub(input2, target2), 2.0), dim=1),
    #               natomsForce)) / 3.0
    return energyRMSE

def calc_mse(input1, input2, target1, target2, natomsEnergy, natomsForce):
    energyRMSE =  torch.sum(torch.pow(torch.div(torch.sub(input1, target1), natomsEnergy),2.))
    forceRMSE = torch.sum(torch.div(torch.sum(torch.pow(torch.sub(input2, target2), 2.0), dim=1), natomsForce)) / 3.0

    return energyRMSE, forceRMSE

class SE(LossFunction):
    def __init__(self, loss_type='SE',
                       cohe = False,
                       energyCoefficient=1.00,
                       forceCoefficient=0.05,
                       device=torch.device("cpu")):
        super(SE, self).__init__(loss_type, cohe, energyCoefficient, forceCoefficient, device)

    def forward(self, input1, input2, target1, target2, natomsEnergy, natomsForce):
        input1 = input1.to(self.device)
        input2 = input2.to(self.device)
        target1 = target1.to(self.device)
        target2 = target2.to(self.device)
        natomsEnergy = natomsEnergy.to(self.device)
        natomsForce = natomsForce.to(self.device)

        if self.cohe:
           #print('natomsF loss', natomsForce)
           self.energyloss =  torch.sum(torch.pow(torch.sub(input1,target1),2.))
           self.forceloss = torch.sum(torch.sum(torch.pow(
                    torch.sub(input2, target2),2.0), dim=1)
                    ) / 3.0
        else:
           self.energyloss =  torch.sum(torch.pow(torch.div(torch.sub(input1,target1), natomsEnergy),2.))
           self.forceloss = torch.sum(torch.div(torch.sum(torch.pow(torch.sub(input2, target2),2.0), dim=1), natomsForce)) / 3.0

        loss = torch.mul(self.energyloss, self.energyCoefficient) + torch.mul(self.forceloss, self.forceCoefficient)
        return loss

class RMSE_noForce(LossFunction):
    '''
    Alan: rewrite of RMSE but with no forces
    '''
    def __init__(self,
                 loss_type='SE',
                 cohe=False,
                 energyCoefficient=1.00,
                 device=torch.device("cpu")):
        super(RMSE_noForce, self).__init__(loss_type, cohe, energyCoefficient,
                                           device)

    def forward(self, input1, target1, natomsEnergy):
        input1 = input1.to(self.device)
        # input2 = input2.to(self.device)
        target1 = target1.to(self.device)
        # target2 = target2.to(self.device)
        natomsEnergy = natomsEnergy.to(self.device)
        # natomsForce = natomsForce.to(self.device)

        if self.cohe:
            #print('natomsF loss', natomsForce)
            self.energyloss = torch.sum(
                torch.pow(torch.sub(input1, target1), 2.))
            # self.forceloss = torch.sum(
            #     torch.sum(torch.pow(torch.sub(input2, target2), 2.0),
            #               dim=1)) / 3.0
        else:
            self.energyloss = torch.sum(
                torch.pow(torch.div(torch.sub(input1, target1), natomsEnergy),
                          2.))

            # self.forceloss = torch.sum(
            #     torch.div(
            #         torch.sum(torch.pow(torch.sub(input2, target2), 2.0),
            #                   dim=1), natomsForce)) / 3.0

        loss = torch.mul(self.energyloss, self.energyCoefficient)
        return loss

class RMLSE(LossFunction):
    def __init__(self, loss_type='RMLSE', 
                 energyCoefficient=1.00, forceCoefficient=0.05, 
                 device=torch.device("cpu")):
        super(RMLSE, self).__init__(loss_type, energyCoefficient, forceCoefficient, device)

    def forward(self, input1, input2, target1, target2, natomsEnergy, natomsForce):
        input1 = input1.to(self.device)
        input2 = input2.to(self.device)
        target1 = target1.to(self.device)
        target2 = target2.to(self.device)
        natomsEnergy = natomsEnergy.to(self.device)
        natomsForce = natomsForce.to(self.device)
        #input2 = torch.div((input2 - self.minF), self.fRange)

        self.energyloss = torch.sum(torch.pow( (torch.log(input1 + 1.) - torch.log(target1 + 1.))/2.303, 2.))
        #self.forceloss = torch.sum( torch.div( torch.sum(
        #                            torch.pow((torch.log(input2 + 1.) - torch.log(target2 + 1.))/2.303, 2.), dim=1), natomsForce)) / 3.0

        self.forceloss = torch.sum(torch.div(torch.sum(torch.pow(
                 torch.sub(input2, target2),2.0), dim=1), natomsForce
                 )) / 3.0
        loss = self.energyloss + torch.mul(self.forceloss, self.force_coefficient)
        return loss
