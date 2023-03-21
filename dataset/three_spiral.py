import numpy as np
import matplotlib.pyplot as plt

# Functionality derived from
# Constructing neural networks for multiclass-discretization based on information entropy
# July 1999IEEE transactions on systems, man, and cybernetics. Part B, Cybernetics:
# a publication of the IEEE Systems, Man, and Cybernetics Society 29(3):445 - 453, page 451
# This function could be generalised to K spirals by letting theta = 2kpi / K where k = K - 1


def spiral(alpha=0.8, noise_scale=0.1, num_points=1000, show=False):

    # Use a function so that different noise values are used for every call
    def noise():
        return noise_scale * np.random.normal(size=num_points)

    # Generate data and labels
    theta = np.linspace(0, 4 * np.pi, num_points)
    rho = alpha * theta

    x1 = rho * np.cos(theta) + noise()
    y1 = rho * np.sin(theta) + noise()
    l1 = np.zeros(num_points)
    d1 = np.stack((x1, y1, l1), axis=-1)

    x2 = rho * np.cos(theta + np.pi) + noise()
    y2 = rho * np.sin(theta + np.pi) + noise()
    l2 = np.ones(num_points)
    d2 = np.stack((x2, y2, l2), axis=-1)

    x3 = rho * np.cos(theta + 2 * np.pi / 3) + noise()
    y3 = rho * np.sin(theta + 2 * np.pi / 3) + noise()
    l3 = np.ones(num_points) * 2
    d3 = np.stack((x3, y3, l3), axis=-1)

    data = np.concatenate((d1, d2, d3), axis=0)

    if show:
        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
        plt.show()

    features = data[:, :2]
    labels = data[:, 2]

    return features, labels
