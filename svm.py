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
def visualise_results(model, logs, extents, num_points):
    # Generate your meshgrid
    x_min, x_max, y_min, y_max = extents
    x, y = np.meshgrid(
        np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points)
    )

    # Parse grid to input that the model can process
    grid = np.stack((x.ravel(), y.ravel()), axis=1)

    # Generate your predictions

    predictions = model.predict(grid)
    predictions = np.reshape(predictions, (num_points, num_points, 2))

    predictions = np.stack(
        (predictions[..., 0], predictions[..., 1], np.zeros(predictions[..., 0].shape)),
        axis=-1,
    )

    labels = np.argmax(predictions[..., :2], axis=2)

    # Plot training and validation history
    plt.figure()
    plt.title("Training vs Validation History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(logs["train_loss"], "r")
    plt.plot(logs["val_loss"], "g")
    plt.legend(["Train loss", "Validation loss"])

    plt.figure()
    plt.title("Training vs Validation History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(logs["train_accs"], "b")
    plt.plot(logs["val_accs"], "y")
    plt.legend(["Train accuracy", "Validation accuracy"])

    plt.figure()
    plt.title("Pixel map")
    plt.scatter(grid[:, 0], grid[:, 1], s=1, c=labels.ravel(), cmap="coolwarm")

    plt.show()


def main():
    # Load dataset
    spiral_ds = SpiralDataset("dataset/spiralsdataset.csv")
    x, y = spiral_ds.x, spiral_ds.y

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=123
    )

    # Create SVM model
    model = SVC()

    # Create parameters
    param_grid = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf", "poly", "sigmoid"],
    }

    # Create grid search to find best parameters
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3)

    # Train model
    grid.fit(x_train, y_train)

    # Get best parameters
    print(grid.best_params_)

    out_map = test(grid, x_test, y_test)
    # # Visualise results
    # plt.scatter(x_test[:, 0], x_test[:, 1], c=grid_predictions)
    # plt.show()


if __name__ == "__main__":
    main()
