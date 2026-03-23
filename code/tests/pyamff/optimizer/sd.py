from torch.optim import Optimizer
from functools import reduce
import torch
import numpy

class SD(Optimizer):
    def __init__(self, params, stepsize=0.01,
                  max_iter=20, max_eval=None,
                  tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
                 ):
        self.stepsize = stepsize
        defaults = dict(max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size)
        super(SD, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                            "(parameter groups)")

        self.params = self.param_groups[0]['params']
        self._numel_cache = None
        self._last_loss = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self.params, 0)
        return self._numel_cache

    def _gather_flat_params(self):
        views = []
        for p in self.params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_grads(self):
        views = []
        for p in self.params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self.params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel()

    def step(self, closure):

        flat_params = self._gather_flat_params().numpy()
        #print('  flat_params before', flat_params)
        #flat_params = torch.from_numpy(flat_params)
        loss = closure()
        self._last_loss = loss
        loss = loss.item()
        flat_grad = self._gather_flat_grads().numpy()
        #print('  loss grads:', loss, flat_grad)
        new_flat_params = flat_params - self.stepsize * numpy.sign(flat_grad)
        #new_flat_params = flat_params - self.stepsize * flat_grad
        #expected = loss + numpy.dot(-self.stepsize * numpy.sign(flat_grad), flat_grad)
        #expected = loss - numpy.dot(self.stepsize * flat_grad, flat_grad)
        #print('  new flat_params for expected', new_flat_params)
        self._distribute_flat_params(torch.from_numpy(new_flat_params))
        return loss


