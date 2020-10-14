import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.von_mises import VonMises

import math

from hyperspherical_vae.distributions.ops import ive


class VonMisesFisher(Distribution):
    """
    The von Mises–Fisher distribution.

    TODO: Perhaps buy the Wood (1994) paper...
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "concentration": constraints.positive,
    }
    support = constraints.real

    def __init__(self, loc: torch.Tensor, concentration: torch.Tensor, vmf_implementation=""):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")

        # For single batches, unsqueeze to (batch_size, dimension) where batch_size = 1.
        if loc.dim() == 1:
            loc = loc.unsqueeze(0)

        self.loc = loc  # Shape: (batch_size, m)
        self.concentration = concentration  # Shape: (batch_size, 1)

        # Distribution is set on the `(self._m - 1)` sphere.
        self._m = self.loc.shape[-1]

        batch_shape = torch.Size([loc.shape[0]])
        event_shape = torch.Size()

        super(VonMisesFisher, self).__init__(
            batch_shape=batch_shape, event_shape=event_shape
        )

    def expand(self, batch_shape, _instance=None):
        pass

    @property
    def mean(self):
        """
        Maximum likelihood estimate for von mises-Fisher distribution, treating `self.loc` as a single sample.
        TODO: The original paper multiplies by A_p(k) instead of dividing; which way is correct?
        https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution#Estimation_of_parameters
        """
        return (
            self.loc
            * ive(self._m / 2 - 1, self.concentration)
            / ive(self._m / 2, self.concentration)
        )

    @property
    def variance(self):
        return None

    def sample(self, sample_shape=torch.Size()):
        """
        TODO: Currently only samples scalar `w`.
        TODO: Move implementation to rsample(), implement full sampling algorithm.
        """
        shape = self._extended_shape(sample_shape)
        w = torch.empty(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self._rejection_sample_prime(self.loc, self.concentration, w)

    def rsample(self, sample_shape=torch.Size()):
        pass

    def log_prob(self, value):
        pass

    def cdf(self, value):
        pass

    def icdf(self, value):
        pass

    def enumerate_support(self, expand=True):
        pass

    def entropy(self):
        pass

    @staticmethod
    def _rejection_sample(
        loc: torch.Tensor, concentration: torch.Tensor, w: torch.Tensor
    ):
        """
        The acceptance-rejection sampling scheme from Ulrich (1984), as implemented
        by Davidson et al. in "Hyperspherical Variational Auto-Encoders."
        """
        batch_size = loc.shape[0]  # also is w.shape[0]
        m = loc.shape[-1]

        b_true = (
            -(2 * concentration) + torch.sqrt(4 * (concentration ** 2) + (m - 1) ** 2)
        ) / (m - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (m - 1) / (4 * concentration)
        s = torch.min(
            torch.max(
                torch.tensor([0.0] * batch_size),
                concentration - 10,
            ),
            torch.tensor([1.0] * batch_size),
        )
        b = b_app * s + b_true * (1 - s)

        a = (
            (m - 1)
            + (2 * concentration)
            + torch.sqrt((4 * (concentration ** 2)) + (m - 1) ** 2)
        ) / 4

        d = (4 * a * b) / (1 + b) - (m - 1) * math.log(m - 1, math.e)

        # Sampling should accept a scalar `w` for each training example.
        done = torch.zeros(w.shape, dtype=torch.bool, device=loc.device)
        while not done.all():
            epsilon = Beta(0.5 * (m - 1), 0.5 * (m - 1)).sample(w.shape)
            w_attempt = (1 - (1 + b) * epsilon) / (
                1 - (1 - b) * epsilon
            )  # TODO: Prime is a better name.
            t = (2 * a * b) / (1 - (1 - b) * epsilon)

            # TODO: One sampled vector v per element in the batch of means.
            u = Uniform(0.0 + 1e-6, 1.0).sample(w.shape)  # TODO: Support batching.

            # TODO: Unit test that accept == accept_prime?
            accept = (m - 1) * torch.log(t) - t + d >= torch.log(u)

            if accept.any():
                w = torch.where(accept, w_attempt, w)
                done = done | accept

        return w

    @staticmethod
    def _rejection_sample_prime(
        loc: torch.Tensor, concentration: torch.Tensor, w: torch.Tensor
    ):
        """
        An alternative acceptance-rejection sampling scheme that also corresponds to Wood (1994)

        Based on TensorFlow's implementation:
        https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/von_mises_fisher.py#L421

        and the implementation from "Spherical Latent Spaces for Stable Variational Autoencoders" by Jiacheng Xu, Greg Durrett
        https://github.com/jiacheng-xu/vmf_vae_nlp/blob/master/NVLL/distribution/vmf_only.py#L92
        """
        m = loc.shape[-1]

        b = (m - 1) / (
            2 * concentration + torch.sqrt((4 * (concentration ** 2)) + (m - 1) ** 2)
        )
        x = (1 - b) / (1 + b)
        c = concentration * x + (m - 1) * torch.log(1 - x ** 2)

        # Sampling should accept a scalar `w` for each training example.
        done = torch.zeros(w.shape, dtype=torch.bool, device=loc.device)
        while not done.all():
            epsilon = Beta(0.5 * (m - 1), 0.5 * (m - 1)).sample(w.shape)
            w_prime = (1 - (1 + b) * epsilon) / (1 - (1 - b) * epsilon)

            # TODO: One sampled vector v per element in the batch of means.
            u = Uniform(0.0 + 1e-6, 1.0).sample(w.shape)

            accept = concentration * w_prime + (m - 1) * torch.log(
                1 - x * w_prime
            ) - c >= torch.log(u)

            if accept.any():
                w = torch.where(accept, w_prime, w)
                done = done | accept

        return w
