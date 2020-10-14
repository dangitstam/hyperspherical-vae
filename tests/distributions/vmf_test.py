import torch
from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform

import math


def test_rejection_sample_methods_equivalent_at_low_dimensions():
    """
    Two implementations exist for Wood (1994)'s acceptance-rejection algorithm. This test shows that these
    implementations, while similar in how often they accept or reject, are not identical.

    Method 1: Davidson et al.'s implementation
    https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/distributions/von_mises_fisher.py#L86

    Method 2: TensorFlow's implementation:
    https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/von_mises_fisher.py#L421
    and the implementation from "Spherical Latent Spaces for Stable Variational Autoencoders"
    by Jiacheng Xu, Greg Durrett
    https://github.com/jiacheng-xu/vmf_vae_nlp/blob/master/NVLL/distribution/vmf_only.py#L92

    """
    batch_size = 8
    shape = torch.Size([batch_size])

    # The number of times in which the two methods differ in acceptance/rejection.
    num_failures = 0

    # Perform trials with hyperspheres of dimensions 2 through 100, over 100 trials.
    for i in range(2, 101):
        for _ in range(0, 101):
            # `batch_size` number of varying means and concentrations.
            loc = torch.rand([batch_size, i])
            concentration = torch.tensor([1, 5, 10, 50.0, 100, 1000, 5000, 10000.0])
            w = torch.empty(shape, dtype=loc.dtype, device=loc.device)
            m = loc.shape[-1]

            # Method 1: Define, b, a, and d, according to Davidson et al.:
            b_true = (
                -(2 * concentration)
                + torch.sqrt(4 * (concentration ** 2) + (m - 1) ** 2)
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

            # Method 2: Define b, x, c, according to Tensorflow:/
            # https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/von_mises_fisher.py#L421
            # and the implementation from "Spherical Latent Spaces for Stable Variational Autoencoders"
            # by Jiacheng Xu, Greg Durrett
            # https://github.com/jiacheng-xu/vmf_vae_nlp/blob/master/NVLL/distribution/vmf_only.py#L92
            b_prime = (m - 1) / (
                2 * concentration
                + torch.sqrt((4 * (concentration ** 2)) + (m - 1) ** 2)
            )
            x_prime = (1 - b_prime) / (1 + b_prime)
            c_prime = concentration * x_prime + (m - 1) * torch.log(1 - x_prime ** 2)

            epsilon = Beta(0.5 * (m - 1), 0.5 * (m - 1)).sample(w.shape)
            w_attempt = (1 - (1 + b) * epsilon) / (1 - (1 - b) * epsilon)
            u = Uniform(0.0 + 1e-6, 1.0).sample(w.shape)

            # Acceptance 1:
            t = (2 * a * b) / (1 - (1 - b) * epsilon)
            accept = (m - 1) * torch.log(t) - t + d >= torch.log(u)

            # Acceptance 2:
            accept_prime = concentration * w_attempt + (m - 1) * torch.log(
                1 - x_prime * w_attempt
            ) - c_prime >= torch.log(u)

            # Assert that they accept the same values for scalar w each time.
            if not (accept == accept_prime).all():
                num_failures += 1

    assert num_failures > 0
