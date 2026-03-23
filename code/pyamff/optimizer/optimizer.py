
class OptimizerWrapper():

    #optimizer = torch.optim.LBFGS(
    #                              self.model.model_params,
    #                              lr = self.model.learningrate, #learning rate
    #                              max_iter = 1,
    #                              max_eval = 1000, #maximal number of function evaluation per optimization
    #                              tolerance_grad = 1e-5,
    #                              tolerance_change = 1e-9,
    #                              history_size = 100,
    #                              line_search_fn = None #either 'strong_wolfe' or None
    #                              )
    optimizer = LBFGSScipy(self.model.model_params, max_iter=1, max_eval=None,
                           tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10)
    #optimizer = torch.optim.SGD(self.model.model_params, lr=self.model.learningrate, momentum=0.9)
    #optimizer = torch.optim.Adam(self.model.model_params, lr= 0.000000001)
    def closure():
        optimizer.zero_grad()
        #(energies, forces) = self.model(atoms_fps, dgdx)
        outputs = self.model(atoms_fps, dgdx)
        #loss = criterion(energies, forces, energy_reference, forces_reference, ntotalAtoms)
        loss = criterion(outputs, targets, ntotalAtoms=30)

        # Used to check if analytical gradient is correct
        #energies.require_grad = True
        #forces.require_grad = True
        #res = torch.autograd.gradcheck(LossFunction(), (energies, forces, energy_reference, forces_reference, ntotalAtoms), raise_exception=False)
        #print('res',res)

        loss.backward()
        # Check average gradient for each layer
        #plot_grad_flow(self.model.model_namedparams)

        # Truncate gradient
        torch.nn.utils.clip_grad_norm_(self.model.model_params, 0.01)
        #torch.nn.utils.clip_grad_value_(self.model.model_params, 0.01)
        return loss
