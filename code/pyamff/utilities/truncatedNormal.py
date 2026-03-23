import torch
import math
from torch.distributions.normal import Normal
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    #traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

class TruncatedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies truncation to a base distribution.
    """

    has_rsample = True

    def __init__(self, base_distribution, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        #super(TruncatedDistribution, self).__init__(*args, **kwargs)
        self.lower_bound, self.upper_bound = broadcast_all(lower_bound, upper_bound)
        self.base_dist = base_distribution
        cdf_low, cdf_high = self.base_dist.cdf(self.lower_bound), self.base_dist.cdf(self.upper_bound)
        self.Z = (cdf_high - cdf_low)

    @constraints.dependent_property
    def arg_constraints(self):
        ac = self.base_dist.arg_constraints.copy()
        ac['lower_bound': constraints.dependent]
        ac['upper_bound': constraints.dependent]
        return ac

    @constraints.dependent_property
    def support(self):
        # Note: The proper way to implement this is intersection([lower_bound, upper_bound], base_dist.support)
        # This requires intersection method to be implemented for constraints.
        return constraints.interval(self.lower_bound, self.upper_bound)

    @property
    def batch_shape(self):
        return self.base_dist.batch_shape

    @property
    def event_shape(self):
        return self.base_dist.event_shape 

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched via inverse cdf sampling. Note that this
        is a generic sampler which is not the most efficient or accurate around tails of base distribution.
        """
        shape = self._extended_shape(sample_shape)
        param = getattr(self.base_dist, list(self.base_dist.params.keys())[0])
        u = param.new(shape).uniform_(torch.finfo(param).tiny, 1 - torch.finfo(param).tiny)
        return self.base_dist.icdf(self.base_dist.cdf(self.lower_bound) + u * self.Z)

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at `value`.
        Returns -inf in value is out of bounds
        """
        log_prob = self.base_dist.log_prob(value)
        log_prob[(value < self.lower_bound) | (value > self.upper_bound)] = -float('inf')
        log_prob = log_prob - self.Z.log()
        return log_prob

    def cdf(self, value):
        """
        Cumulative distribution function for the truncated distribution
        """
        return (self.base_dist.cdf(value) - self.base_dist.cdf(self.lower_bound)) / self.Z


def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    is_scalar = False
    if x.ndim == 0:
        x = x.view(1)
        is_scalar = True
        sample_shape = torch.Size([1])

    #done = torch.zeros(sample_shape).byte()
    done = torch.zeros(sample_shape, dtype=torch.bool)
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x**2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound**2 - proposed_x**2)
        else:  # both positive
            if isinstance(lower_bound, torch.Tensor):
                # assert(lower_bound.gt(0.0).all()) # Alan: bug fix
                pass
            else:
                 # assert(lower_bound > 0.0)
                 pass
            log_prob_accept = 0.5 * (lower_bound**2 - proposed_x**2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        #accept = torch.bernoulli(prob_accept).byte() & ~done
        accept = torch.bernoulli(prob_accept).bool() & ~done
        if accept.any():
            x[accept] = proposed_x[accept]
            done |= accept
    
    if is_scalar:
        return x.view(())
    return x


class TruncatedNormal(TruncatedDistribution):
    r"""
    Creates a normal distribution parameterized by
    `loc` and `scale`, bounded to [`lower_bound`, `upper_bound`]
    Example::
        >>> m = TruncatedNormal(torch.Tensor([0.0]), torch.Tensor([1.0]), torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # normal distributed with mean=0 and stddev=1, bounded to [0,1]
         0.1046
        [torch.FloatTensor of size 1]
    Args:
        loc (Tensor): mean of the distribution
        scale (Tensor): standard deviation of the distribution
        lower_bound (Tensor): lower bound for the distribution. Best to keep it greater than loc-4*scale for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than loc+4*scale for
        stable results
    """

    def __init__(self, loc, scale, lower_bound=-float('inf'), upper_bound=float('inf'), *args, **kwargs):
        super().__init__(Normal(loc, scale), lower_bound, upper_bound, *args, **kwargs)
        # TODO: default values for lower and upper bounds

    def rsample(self, sample_shape=torch.Size()):
        eps = _standard_truncnorm_sample((self.lower_bound - self.base_dist.mean) / self.base_dist.stddev,
                                              (self.upper_bound - self.base_dist.mean) / self.base_dist.stddev,
                                              sample_shape=sample_shape)
        return eps * self.base_dist.stddev + self.base_dist.mean

    @property
    def mean(self):
        return (self.base_dist.mean +
                self.base_dist.stddev *
                (self.base_dist.log_prob(self.lower_bound).exp() -
                 self.base_dist.log_prob(self.upper_bound).exp()) / self.Z)

    @property
    def variance(self):
        return self.base_dist.variance * (
            1 +
            (self.lower_bound * self.base_dist.log_prob(self.lower_bound).exp() -
             self.upper_bound * self.base_dist.log_prob(self.upper_bound).exp()
             ) / self.Z -
            (self.base_dist.log_prob(self.lower_bound).exp() -
             self.base_dist.log_prob(self.upper_bound).exp()
             )**2 / self.Z**2
            )

    @property
    def stddev(self):
        return self.variance ** 0.5

    @property
    def entropy():
        return (0.5 * math.log(2 * math.pi) + self.stddev.log() + self.Z.log() +
                0.5 * (self.lower_bound * self.base_dist.log_prob(self.lower_bound).exp() -
                       self.upper_bound * self.base_dist.log_prob(self.upper_bound).exp()
                       ) / self.Z)
