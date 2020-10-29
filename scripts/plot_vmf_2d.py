import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from hyperspherical_vae.distributions.vmf import VonMisesFisher


def main():
    sns.set_theme()
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.gca().set_aspect("equal", adjustable="box")

    mean_1 = torch.tensor([1.0, 0.0])
    mean_2 = torch.tensor([-math.sqrt(2) / 2, -math.sqrt(2) / 2])
    mean_3 = torch.tensor([1 / 2, math.sqrt(3) / 2])

    vmf_1 = VonMisesFisher(mean_1, torch.tensor([50.0]))
    vmf_2 = VonMisesFisher(mean_2, torch.tensor([50.0]))
    vmf_3 = VonMisesFisher(mean_3, torch.tensor([50.0]))

    for i in range(50):
        x_1, y_1 = vmf_1.sample().squeeze().tolist()
        plt.scatter(x_1, y_1, color="r", marker=".")

        x_2, y_2 = vmf_2.sample().squeeze().tolist()
        plt.scatter(x_2, y_2, color="g", marker=".")

        x_3, y_3 = vmf_3.sample().squeeze().tolist()
        plt.scatter(x_3, y_3, color="m", marker=".")

    plt.show()


if __name__ == "__main__":
    main()
