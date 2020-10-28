import torch

from hyperspherical_vae.distributions.vmf import VonMisesFisher
import torch.nn.functional as F


class NVDM(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        TODO: Parameters for m (# of dimensions), type of reconstruction loss (binary cross entropy, MSE).
        """
        super(NVDM, self).__init__()

        # TODO: Keep it simple for now, activations and layering later.
        self.mean_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
        )
        self.concentration_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            # TODO: Concentration always needs to be non-negative; is this the way to do it?
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, input_dim),
            # TODO: BCELoss requires non-negative scalar inputs; is this the way to do it?
            torch.nn.ReLU()
        )

        self.reconstruction_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        """"""
        # Project input into a mean for the vMF. Ensure each mean is a unit vector.
        mean = self.mean_encoder(x)  # Shape: (batch_size, hidden_dim)
        mean /= mean.norm(dim=-1).unsqueeze(-1).repeat(1, mean.shape[-1])  # Shape: (batch_size, hidden_dim).

        # Concentration needs to be non-negative for numerical stability in computation of KL-divergence.
        # More specifically, since the log modified Bessel is used, the instability is introduced when
        # log(Iv(m/2, 0)) = log(0). This also prevents collapsing into the uniform prior.
        concentration = self.concentration_encoder(x) + 1  # Shape: (batch_size,)

        vmf = VonMisesFisher(mean, concentration)

        z = vmf.rsample()

        x_prime = self.decoder(z)

        loss = vmf.kl_divergence().mean(-1) + self.reconstruction_loss(x_prime, x)

        return {
            "concentration": mean,
            "loss": loss,
            "mean": mean,
            "reconstruction": x_prime,  # TODO: Batchnorm?
            "z": z,
        }
