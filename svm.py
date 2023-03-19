import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from dataset.spiral_ds_loader import SpiralDataset

# Compute final prediction for test set
def test(model, x_test, y_test):

    # Use model to predict on test set
    y_pred_test = model.predict(x_test)

    # Get confusion matrix, accuracy, precision and recall
    confusion_mat = confusion_matrix(y_test, y_pred_test)

    acc = accuracy_score(y_test, y_pred_test)

    precision_global = precision_score(y_test, y_pred_test, average="micro")
    precision_mean = precision_score(y_test, y_pred_test, average="macro")

    recall_global = recall_score(y_test, y_pred_test, average="micro")
    recall_mean = recall_score(y_test, y_pred_test, average="macro")

    out_map = {
        "conf_mat": confusion_mat,
        "acc": acc,
        "precision_global": precision_global,
        "precision_mean": precision_mean,
        "recall_global": recall_global,
        "recall_mean": recall_mean,
    }

    return out_map


# Function for visualising model
def visualise_results(model, extents, num_points):
    # Generate meshgrid
    x_min, x_max, y_min, y_max = extents
    x, y = np.meshgrid(
        np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points)
    )

    # Parse grid to input that the model can process
    grid = np.stack((x.ravel(), y.ravel()), axis=1)

    # Generate predictions
    predictions = model.predict(grid)

    plt.figure()
    plt.title("Pixel map")
    plt.scatter(grid[:, 0], grid[:, 1], s=1, c=predictions, cmap="coolwarm")

    plt.show()


def main():
    # Load dataset
    spiral_ds = SpiralDataset("dataset/spiralsdataset.csv")
    x, y = spiral_ds.x, spiral_ds.y

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=123
    )

    # Create parameters
    param_grid = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf", "sigmoid"],
    }

    # Create grid search to find best parameters
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # Train model
    grid.fit(x_train, y_train)

    out_map = test(grid, x_test, y_test)

    print("\nConfusion matrix: \n{}\n".format(out_map["conf_mat"]))
    print(" Accuracy - {:.4f}".format(out_map["acc"]))
    print(
        "Precision - Global: {:.4f} \t Mean: {:.4f}".format(
            out_map["precision_global"], out_map["precision_mean"]
        )
    )
    print(
        "   Recall - Global: {:.4f} \t Mean: {:.4f}".format(
            out_map["recall_global"], out_map["recall_mean"]
        )
    )

    # Visualise results
    visualise_results(grid, extents=[-6, 6, -6, 6], num_points=1000)


if __name__ == "__main__":
    main()
