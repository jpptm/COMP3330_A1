import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from numpy import pi
from torch.utils.data import Dataset
import torch
import pandas as pd


############################ Debugging script for learning curves ##########################################


# N spiral dataset
# def spiral(alpha=6, noise_scale=0.01, num_points=75, n_spirals=2, show=False):

#     # Use a function so that different noise values are used for every call
#     def noise():
#         return noise_scale * np.random.normal(size=num_points)

#     # Generate data and labels
#     theta = np.linspace(0, 1, num_points)
#     rho = alpha * theta

#     data = []

#     for k in range(1, n_spirals + 1):
#         x = rho * np.cos((2 * pi / n_spirals) * (2 * theta + k - 1 + noise()))
#         y = rho * np.sin((2 * pi / n_spirals) * (2 * theta + k - 1 + noise()))
#         label = (k - 1) * np.ones(num_points)
#         d = np.stack((x, y, label), axis=-1)
#         data.append(d)

#     data = np.array(data).reshape(-1, 3)

#     cmap = {0: "red", 1: "blue", 2: "green"}
#     colors = [cmap[i] for i in data[:, 2]]
#     if show:
#         plt.scatter(data[:, 0], data[:, 1], c=colors)
#         plt.show()

#     return data[..., :2], data[..., 2]


# Add csv loaderclass
class SpiralDataset(Dataset):
    def __init__(self, csv_file, extra_features=False):
        self.data = pd.read_csv(csv_file, header=None).values
        self.x = torch.tensor(self.data[:, :2], dtype=torch.float32)
        self.y = torch.tensor(self.data[:, 2], dtype=torch.int64)

        # If extra features is true, add additional features to the dataset and train on this
        if extra_features:
            # Features to add: radius, theta, x1x2,
            r = torch.linalg.norm(self.x, dim=-1)
            theta = torch.atan2(self.x[:, 1], self.x[:, 0])
            x1x2 = self.x[:, 0] * self.x[:, 1]

            self.x = torch.stack([self.x[:, 0], self.x[:, 1], r, theta, x1x2], dim=-1)

    def __len__(self):
        return len(self.data)

    # Fill this member so we can get data by index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Generate a 2-spiral dataset
spiral = SpiralDataset("dataset/spiralsdataset.csv")
X, y = spiral.x, spiral.y

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", edgecolors="b")
# Split the dataset into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the SVM model
svm_model = SVC(kernel="rbf", gamma=0.7, C=10)

# Define the learning curve function
def plot_learning_curve(X, y, model):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")

    return plt


# Plot the learning curve
plot_learning_curve(X_train, y_train, svm_model)

plt.show()
