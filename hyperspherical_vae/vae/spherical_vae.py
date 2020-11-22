import torch

from hyperspherical_vae.distributions.vmf import VonMisesFisher


class SphericalVAE(torch.nn.Module):  # TODO: Rename this class.
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int = 2):
        """
        TODO: Parameters for m (# of dimensions), type of reconstruction loss (binary cross entropy, MSE).
        """
        super(SphericalVAE, self).__init__()

        self.initial_latent_projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
        )

        # TODO: Keep it simple for now, activations and layering later.
        self.mean_encoder = torch.nn.Sequential(torch.nn.Linear(hidden_dim, latent_dim))
        self.concentration_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            # TODO: Concentration always needs to be non-negative; is this the way to do it?
            torch.nn.GELU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, input_dim),
        )

        self.reconstruction_loss = torch.nn.L1Loss()

    def forward(self, x: torch.Tensor):
        """"""
        x_initial_projection = self.initial_latent_projection(x)

        # Project input into a mean for the vMF. Ensure each mean is a unit vector.
        mean_prime = self.mean_encoder(
            x_initial_projection
        )  # Shape: (batch_size, hidden_dim)
        mean = mean_prime / (
            mean_prime.norm(dim=-1).unsqueeze(-1).repeat(1, mean_prime.shape[-1])
        )  # Shape: (batch_size, hidden_dim).

        # Concentration needs to be non-negative for numerical stability in computation of KL-divergence.
        # More specifically, since the log modified Bessel is used, the instability is introduced when
        # log(Iv(m/2, 0)) = log(0). This also prevents collapsing into the uniform prior.
        concentration = (
            self.concentration_encoder(x_initial_projection) + 1
        )  # Shape: (batch_size,)

        vmf = VonMisesFisher(mean, concentration)

        z = vmf.rsample()

        x_prime = self.decoder(z)

        # TODO: Return loss in two parts.
        loss = vmf.kl_divergence().mean() + self.reconstruction_loss(x_prime, x)

        return {
            "concentration": concentration,
            "loss": loss,
            "mean": mean,
            "reconstruction": x_prime,  # TODO: Batchnorm?
            "z": z,
        }
