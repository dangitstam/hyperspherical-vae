import math

from hyperspherical_vae.vae import NVDM
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from hyperspherical_vae.distributions.vmf import VonMisesFisher

from torch.utils.data import DataLoader


NUM_SAMPLES = 500


def create_noisy_nonlinear_transformation(input_dim: int, target_dim: int):
    """
    Given a tensor of size (hidden_dim, target_dim), returns a function that given a tensor x,
    produces a noisy, nonlinear transformation x' of size (batch_size, target_dim).
    """
    projection = torch.randn(input_dim, target_dim)

    # Gaussian noise is added to the input before projection to the target dim.
    # Each example will have it's own distinct noise.
    # The reciprocal serves as the nonlinear transformation.
    return lambda x: 1 / torch.matmul(x + torch.randn(x.size()), projection)


def training_epoch(nvdm: NVDM, training_dataloader: DataLoader):
    # Iterate over (batch_size, input_dim) examples.
    for example in iter(training_dataloader):
        pass


def train(nvdm: NVDM, training_dataloader: DataLoader):
    pass


def main():
    sns.set_theme()
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.gca().set_aspect("equal", adjustable="box")

    mean_1 = torch.tensor([math.sqrt(3) / 2, -1 / 2])
    mean_2 = torch.tensor([-math.sqrt(2) / 2, -math.sqrt(2) / 2])
    mean_3 = torch.tensor([-1 / 2, math.sqrt(3) / 2])

    vmf_1 = VonMisesFisher(mean_1, torch.tensor([5.0]))
    vmf_2 = VonMisesFisher(mean_2, torch.tensor([10.0]))
    vmf_3 = VonMisesFisher(mean_3, torch.tensor([2.0]))

    # TODO: Move this to a docstring.
    # A VAE with a hidden dim of 2 will be trained on 100-dimensional inputs
    # created via noisy,nonlinear transformation of the 2-dimensional vMF samples.
    # At evaluation time, the VAE will be given additional samples and the
    # 2-dimensional latent vectors will be plotted to verify that the VAE can learn
    # a circular latent space.
    noisy_nonlinear_transformation = create_noisy_nonlinear_transformation(2, 100)
    training_data = []

    for i in range(NUM_SAMPLES):
        sample_1 = vmf_1.sample()
        x_1, y_1 = sample_1.squeeze().tolist()
        plt.scatter(x_1, y_1, color="r", marker=".")

        sample_2 = vmf_2.sample()
        x_2, y_2 = sample_2.squeeze().tolist()
        plt.scatter(x_2, y_2, color="g", marker=".")

        sample_3 = vmf_3.sample()
        x_3, y_3 = sample_3.squeeze().tolist()
        plt.scatter(x_3, y_3, color="m", marker=".")

        training_data.append(noisy_nonlinear_transformation(sample_1.squeeze()))
        training_data.append(noisy_nonlinear_transformation(sample_2.squeeze()))
        training_data.append(noisy_nonlinear_transformation(sample_3.squeeze()))

    training_dataloader = DataLoader(
        training_data, batch_size=4, shuffle=True, num_workers=4
    )

    plt.show()


if __name__ == "__main__":
    main()
