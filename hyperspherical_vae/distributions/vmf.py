import math

import torch
from torch.distributions import constraints
from torch.distributions.beta import Beta
from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform

from hyperspherical_vae.distributions.ops import ive


class VonMisesFisher(Distribution):
    """
    The von Mises–Fisher distribution.

    TODO: Enable multiple implementations (e.g. choose whether to use householder).
    TODO: Perhaps buy the Wood (1994) paper...
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "concentration": constraints.positive,
    }
    support = constraints.real

    def __init__(
        self,
        loc: torch.Tensor,
        concentration: torch.Tensor,
        change_magnitude_sampling_algorithm: str = "wood",
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")

        if concentration.dim() > 2 or (
            concentration.dim() == 2 and concentration.shape[-1] != 1
        ):
            raise ValueError(
                """
                `concentration` should be a tensor of a single value with shape (1,) 
                or batched with shapes (batch_size,) or (batch_size, 1); got {} instead
                """.format(
                    concentration.size()
                )
            )

        # TODO: Some torch distributions will repeat a parameter like this if only one is defined.
        if loc.shape[0] != concentration.shape[0]:
            raise ValueError(
                """
                batch size for loc ({}) and concentration ({}) differ; 
                concentration should be defined for each mean
                """.format(
                    loc.shape[0], concentration.shape[0]
                )
            )

        # Invariant: `self.concentration` should always have the shape (batch_size,).
        # Feedforward layers may project to a single dimension and produce shape (batch_size, 1).
        # Computing batched latent representations (w; sqrt(1 - w^t) v.T)^T however requires (batch_size,) for proper
        # matrix multiply.
        if concentration.dim() > 1:
            concentration = concentration.squeeze(-1)

        if change_magnitude_sampling_algorithm.lower() not in ("wood", "ulrich"):
            raise ValueError(
                "unsupported change magnitude sampling algorithm: {}".format(
                    change_magnitude_sampling_algorithm
                )
            )

        # For single batches, unsqueeze to (batch_size, dimension) where batch_size = 1.
        if loc.dim() == 1:
            loc = loc.unsqueeze(0)

        self.loc = loc  # Shape: (batch_size, m)
        self.concentration = concentration  # Shape: (batch_size,)

        change_magnitude_sampling_algorithms = {
            "wood": self._rejection_sample_wood,
            "ulrich": self._rejection_sample_ulrich,
        }
        self._rejection_sample = change_magnitude_sampling_algorithms[
            change_magnitude_sampling_algorithm
        ]

        # Distribution is set on the `(self._m - 1)` sphere.
        self._m = self.loc.shape[-1]

        batch_shape = loc.shape
        event_shape = torch.Size()

        super(VonMisesFisher, self).__init__(
            batch_shape=batch_shape, event_shape=event_shape
        )

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

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """
        TODO: How should `sample_shape` be handled?
        """
        if sample_shape != torch.Size():
            raise ValueError(
                "sample_shape {} unsupported, currently only batched predictions are supported"
            )

        shape = self._extended_shape(sample_shape)  # Shape: (batch_size, m).
        batch_size = shape[0]

        # Batched modal vector ((1,0,...,0) repeated batch_size-number of times).
        e1 = torch.zeros(shape)  # Shape: (batch_size, m).
        e1.T[0] = 1

        # Sample vector v ~ U(S^(m - 2)) by sampling (batch_size, m - 1) values
        # from a Gaussian and normalize each (m - 1)-sized vector (Muller 1959, Marsaglia 1972).
        v = torch.randn(batch_size, self._m - 1)  # Shape: (batch_size, self._m - 1)
        v_norm = (
            v.norm(dim=-1).unsqueeze(-1).repeat(1, self._m - 1)
        )  # Shape: (batch_size, self._m - 1)
        v /= v_norm

        w = torch.empty(
            torch.Size([batch_size]), dtype=self.loc.dtype, device=self.loc.device
        )  # Shape: (batch_size,)
        w = self._rejection_sample(
            self.loc, self.concentration, w
        )  # Shape: (batch_size,)

        # Sample z' with modal vector e1 = (w; sqrt(1 - w^2) v^T)^T
        # Shape: (batch_size, m)
        z_prime = torch.cat([w.unsqueeze(0), torch.sqrt((1 - w ** 2)) * v.T]).T

        # Shape: (batch_size, m, m)
        householder_transform = self._householder_transform(self.loc, e1)

        # Shape: (batch_size, m).
        return torch.matmul(householder_transform, z_prime.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, value: torch.Tensor):
        return self._log_prob_unnormalized(value) + self._log_prob_normalization()

    def kl_divergence(self):
        """
        The KL-divergence KL(q(z | u, k) || U(S^(m - 1))) of the von-Mises Fisher
        against the uniform distribution on the m-dimensional unit hypersphere,
        as derived by Davidson et al. in "Hyperspherical Variational Auto-Encoders."
        """
        return (
            (
                self.concentration
                * ive(self._m / 2, self.concentration)
                / ive(self._m / 2 - 1, self.concentration)
            )
            + self._log_prob_normalization()
            # Subtracting the log surface area of the hypersphere.
            + (
                (self._m / 2) * math.log(math.pi)
                + math.log(2)
                - torch.lgamma(torch.Tensor([self._m / 2]))
            )
        )

    @staticmethod
    def _rejection_sample_ulrich(
        loc: torch.Tensor, concentration: torch.Tensor, w: torch.Tensor
    ):
        """
        The acceptance-rejection sampling scheme from Ulrich (1984), as implemented
        by Davidson et al. in "Hyperspherical Variational Auto-Encoders."
        """
        batch_size = loc.shape[0]
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
            w_prime = (1 - (1 + b) * epsilon) / (1 - (1 - b) * epsilon)
            t = (2 * a * b) / (1 - (1 - b) * epsilon)

            u = Uniform(0.0 + 1e-6, 1.0).sample(w.shape)

            accept = (m - 1) * torch.log(t) - t + d >= torch.log(u)

            if accept.any():
                w = torch.where(accept, w_prime, w)
                done = done | accept

        return w

    @staticmethod
    def _rejection_sample_wood(
        loc: torch.Tensor, concentration: torch.Tensor, w: torch.Tensor
    ):
        """
        The acceptance-rejection sampling scheme from Wood (1994).

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

            u = Uniform(0.0 + 1e-6, 1.0).sample(w.shape)

            accept = concentration * w_prime + (m - 1) * torch.log(
                1 - x * w_prime
            ) - c >= torch.log(u)

            if accept.any():
                w = torch.where(accept, w_prime, w)
                done = done | accept

        return w

    @staticmethod
    def _householder_transform(mean: torch.Tensor, e1: torch.Tensor):
        """
        The Householder transform.

        Given that `mean` is the mean of a von-Mises Fisher distribution, results in an orthogonal transformation that
        rotates a sample z' ~ q(z | `e1`, k) such that the rotated sample is distributed according to q(z | `mean`, k).
        """
        if mean.dim() < 1:
            raise ValueError("mean must be at least one-dimensional.")

        batch_size = mean.shape[0]
        m = mean.shape[-1]

        mean_prime = e1 - mean  # Shape: (batch_size, m)
        mean_prime_norm = (
            mean_prime.norm(dim=-1).unsqueeze(-1).repeat(1, m)
        )  # Shape: (batch_size, m)
        mean_prime /= mean_prime_norm  # Shape: (batch_size, m)

        # Shape: (batch_size, m, m)
        batch_identity_matrix = torch.diag(torch.ones(m)).repeat(batch_size, 1, 1)

        # Shape: (batch_size, m, m)
        householder_transform = batch_identity_matrix - 2 * torch.matmul(
            mean.unsqueeze(-1),  # Shape: (batch_size, m, 1)
            mean.unsqueeze(-1).permute(0, 2, 1),  # Shape: (batch_size, 1, m)
        )

        return householder_transform

    def _log_prob_unnormalized(self, x: torch.Tensor):
        return (
            self.concentration.unsqueeze(-1).repeat(1, self._m)
            * self.loc  # Shape: (batch_size, m)
            * x  # Shape: (batch_size, m)
        ).sum(
            -1
        )  # Shape: (batch_size,)

    def _log_prob_normalization(self):
        return (
            (self._m / 2 - 1) * torch.log(self.concentration)
            - (self._m / 2) * math.log(2 * math.pi)
            # Note: log(Iv(x)) = x + log(e^(-x) * Iv(x))
            - (self.concentration + torch.log(ive(self._m / 2 - 1, self.concentration)))
        )
