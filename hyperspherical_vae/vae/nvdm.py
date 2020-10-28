import torch

from hyperspherical_vae.distributions.vmf import VonMisesFisher


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
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, input_dim)
        )

        self.reconstruction_loss = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        """"""
        # TODO: Ensure each mean is a unit vector.
        mean = self.mean_encoder(x)  # Shape: (batch_size, m)

        concentration = self.concentration_encoder(x)  # Shape: (batch_size,)

        vmf = VonMisesFisher(mean, concentration)

        z = vmf.rsample()

        x_prime = self.decoder(z)

        # TODO: Sum to a single scalar.
        loss = vmf.kl_divergence() + self.reconstruction_loss(x, x_prime)

        return {
            "concentration": mean,
            "loss": loss,
            "mean": mean,
            "reconstruction": x_prime,  # TODO: Batchnorm?
            "z": z,
        }
