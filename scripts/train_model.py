import math

from tqdm import tqdm

from hyperspherical_vae.vae import SphericalVAE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim

from hyperspherical_vae.distributions.vmf import VonMisesFisher

from torch.utils.data import DataLoader


NUM_SAMPLES = 100
BATCH_SIZE = 5
NUM_EPOCHS = 10


def create_noisy_nonlinear_transformation(input_dim: int, target_dim: int):
    """
    Given a tensor of size (hidden_dim, target_dim), returns a function that given a tensor x,
    produces a noisy, nonlinear transformation x' of size (batch_size, target_dim).
    """
    projection = torch.randn(input_dim, target_dim)

    # Gaussian noise is added to the input before projection to the target dim.
    # Each example will have it's own distinct noise.
    # The reciprocal serves as the nonlinear transformation.
    return lambda x: torch.matmul(x, projection)


def training_epoch(model: SphericalVAE, training_dataloader: DataLoader):
    # Iterate over (batch_size, input_dim) examples.
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    t = tqdm(iter(training_dataloader))
    total_loss = 0
    for example in t:
        optimizer.zero_grad()
        output = model(example)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.data

        t.set_description("loss: {}".format(loss.data))

    print(total_loss / NUM_SAMPLES)


def main():
    """
    A VAE with a hidden dim of 2 will be trained on 100-dimensional inputs
    created via noisy,nonlinear transformation of the 2-dimensional vMF samples.

    At evaluation time, the VAE will be given additional samples and the
    2-dimensional latent vectors will be plotted to verify that the VAE can learn
    a circular latent space.
    """
    torch.autograd.set_detect_anomaly(True)

    mean_1 = torch.tensor([math.sqrt(3) / 2, -1 / 2])
    mean_2 = torch.tensor([-math.sqrt(2) / 2, -math.sqrt(2) / 2])
    mean_3 = torch.tensor([-1 / 2, math.sqrt(3) / 2])

    vmf_1 = VonMisesFisher(mean_1, torch.tensor([5.0]))
    vmf_2 = VonMisesFisher(mean_2, torch.tensor([10.0]))
    vmf_3 = VonMisesFisher(mean_3, torch.tensor([2.0]))

    noisy_nonlinear_transformation = create_noisy_nonlinear_transformation(2, 100)
    training_data = []

    for i in tqdm(range(NUM_SAMPLES)):
        sample_1 = vmf_1.sample()
        sample_2 = vmf_2.sample()
        sample_3 = vmf_3.sample()

        training_data.append(noisy_nonlinear_transformation(sample_1.squeeze()))
        training_data.append(noisy_nonlinear_transformation(sample_2.squeeze()))
        training_data.append(noisy_nonlinear_transformation(sample_3.squeeze()))

    model = SphericalVAE(100, 25, 2)
    training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(NUM_EPOCHS):
        training_epoch(model, training_dataloader)

    def plot_latent_representation_of_noisy_samples(
        vmf: VonMisesFisher, model: SphericalVAE, ax, color: str, num_samples=200
    ):
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        average_mean = torch.zeros(2).unsqueeze(0)
        average_concentration = torch.zeros(1).unsqueeze(0)
        for _ in tqdm(range(num_samples)):
            sample = vmf.sample()
            sample_transformed = noisy_nonlinear_transformation(sample)
            output = model(sample_transformed)
            average_mean += output["mean"]
            average_concentration += output["concentration"]
            x, y = output["z"].squeeze().tolist()
            ax.scatter(x, y, color=color, marker=".")

        average_mean /= num_samples
        average_concentration /= num_samples

        ax.text(-1.5, 1.5, "Average mean: {}".format(average_mean.squeeze().data))
        ax.text(
            -1.5,
            -1.5,
            "Average concentration: {}".format(average_concentration.squeeze().data),
        )

    sns.set_theme()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221, adjustable="box", aspect=1.0)
    ax2 = fig.add_subplot(222, adjustable="box", aspect=1.0)
    ax3 = fig.add_subplot(223, adjustable="box", aspect=1.0)
    plot_latent_representation_of_noisy_samples(vmf_1, model, ax1, "r")
    plot_latent_representation_of_noisy_samples(vmf_2, model, ax2, "g")
    plot_latent_representation_of_noisy_samples(vmf_3, model, ax3, "m")

    ax4 = fig.add_subplot(224, adjustable="box", aspect=1.0)
    ax4.set_xlim([-2, 2])
    ax4.set_ylim([-2, 2])
    for _ in tqdm(range(200)):
        sample_1 = vmf_1.sample()
        sample_2 = vmf_2.sample()
        sample_3 = vmf_3.sample()

        sample_1_transformed = noisy_nonlinear_transformation(sample_1)
        sample_2_transformed = noisy_nonlinear_transformation(sample_2)
        sample_3_transformed = noisy_nonlinear_transformation(sample_3)

        output_1 = model(sample_1_transformed)
        output_2 = model(sample_2_transformed)
        output_3 = model(sample_3_transformed)

        x_1, y_1 = output_1["z"].squeeze().tolist()
        x_2, y_2 = output_2["z"].squeeze().tolist()
        x_3, y_3 = output_3["z"].squeeze().tolist()

        ax4.scatter(x_1, y_1, color="r", marker=".")
        ax4.scatter(x_2, y_2, color="g", marker=".")
        ax4.scatter(x_3, y_3, color="m", marker=".")

    plt.show()


if __name__ == "__main__":
    main()
