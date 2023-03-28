import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

# Functionality derived from
# Constructing neural networks for multiclass-discretization based on information entropy
# July 1999IEEE transactions on systems, man, and cybernetics. Part B, Cybernetics:
# a publication of the IEEE Systems, Man, and Cybernetics Society 29(3):445 - 453, page 451
# and https://atcold.github.io/pytorch-Deep-Learning/en/week02/02-3/


class NSpiralDataset(Dataset):
    def __init__(
        self, alpha=6, noise_scale=0.01, num_points=50, n_spirals=3, show=False
    ):  # NB: the length of the dataset will be num_points * n_spirals

        self.data = self.spiral(
            alpha=alpha,
            noise_scale=noise_scale,
            num_points=num_points,
            n_spirals=n_spirals,
            show=show,
        )

        self.x = torch.tensor(self.data[:, :2], dtype=torch.float32)
        self.y = torch.tensor(self.data[:, 2], dtype=torch.int64)

    def spiral(self, alpha, noise_scale, num_points, n_spirals, show):

        # Use a function so that different noise values are used for every call
        def noise():
            return noise_scale * np.random.normal(size=num_points)

        # Generate data and labels
        theta = np.linspace(0, 1, num_points)
        rho = alpha * theta

        data = []

        for k in range(1, n_spirals + 1):
            x = rho * np.cos((2 * pi / n_spirals) * (2 * theta + k - 1 + noise()))
            y = rho * np.sin((2 * pi / n_spirals) * (2 * theta + k - 1 + noise()))
            label = (k - 1) * np.ones(num_points)
            d = np.stack((x, y, label), axis=-1)
            data.append(d)

        data = np.array(data).reshape(-1, 3)

        cmap = {0: "red", 1: "blue", 2: "green"}
        colors = [cmap[i] for i in data[:, 2]]
        if show:
            plt.scatter(data[:, 0], data[:, 1], c=colors)
            plt.show()

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
