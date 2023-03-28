import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from dataset.spiral_ds_loader import SpiralDataset
from dataset.three_spiral import NSpiralDataset


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


# Visualise likelihood of overfitting or underfitting
def plot_learning_curve(x_src, y_src, gamma_val):
    model = SVC(gamma=gamma_val)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(
        model, x_src, y_src, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=4
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(f"Learning curve for gamma={gamma_val}")
    plt.xlabel("Training examples")
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


# Function for visualising model
def visualise_results(
    model,
    gamma,
    extents,
    num_points,
    x_src,
    y_src,
    test_size,
    x_test,
    y_test,
    colors=["red", "blue"],
):
    # Generate meshgrid
    x_min, x_max, y_min, y_max = extents
    x, y = np.meshgrid(
        np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points)
    )

    # Parse grid to input that the model can process
    grid = np.stack((x.ravel(), y.ravel()), axis=1)

    # Generate predictions
    predictions = model.predict(grid)
    cmap = {0: "red", 1: "blue", 2: "green"}
    colors = [cmap[i] for i in predictions]

    plt.figure()
    plt.title("Pixel map")
    plt.scatter(grid[:, 0], grid[:, 1], s=1, c=colors, cmap="coolwarm")

    # Predict probabilities for the test set
    probas = model.predict_proba(x_test)

    # Calculate the false positive rate, true positive rate, and thresholds for each class
    results = {"fpr": [], "tpr": [], "roc_auc": []}
    for i in range(len(np.unique(y_test))):
        fpr, tpr, _ = roc_curve(y_test, probas[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)

        results["fpr"].append(fpr)
        results["tpr"].append(tpr)
        results["roc_auc"].append(roc_auc)

    # Plot the ROC curves for each class
    plt.figure()
    for i, color in zip(range(len(np.unique(y_test))), ["red", "blue", "green"]):
        plt.plot(
            results["fpr"][i],
            results["tpr"][i],
            lw=1,
            alpha=1,
            color=color,
            label="ROC class %d (AUC = %0.2f)" % (i, results["roc_auc"][i]),
        )

    # Plot the random line for comparison
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random")

    # Set the plot title and axis labels
    plt.title("Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Set the axis limits and legend
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")

    # Plot the learning curve
    plot_learning_curve(x_src, y_src, gamma)

    plt.show()


def main(C, gamma, test_size, auto_grid=False):
    # Load dataset
    # spiral_ds = SpiralDataset("dataset/spiralsdataset.csv")
    spiral_ds = NSpiralDataset()
    x, y = spiral_ds.x, spiral_ds.y
    print(len(x))

    # Show dataset
    plt.figure()
    cmap = {0: "red", 1: "blue", 2: "green"}
    colors = [cmap[i] for i in y.numpy()]
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    # plt.show()

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=123, shuffle=True
    )

    grid = SVC(kernel="rbf", C=C, gamma=gamma, probability=True)

    if auto_grid:
        # Create parameters
        param_grid = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "kernel": ["rbf", "sigmoid"],
        }

        # Create grid search to find best parameters and show the best parameters
        grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3)

    # Train model
    train_begin = time.time()
    grid.fit(x_train, y_train)
    print(f"Training time: {time.time() - train_begin:.2f} seconds")

    print("\nEvaluating model on the test set...")

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
    print("\nVisualising results...")
    input_map = {
        "model": grid,
        "gamma": grid.best_params_["gamma"] if auto_grid else gamma,
        "extents": [-6, 6, -6, 6],
        "num_points": 1000,
        "x_src": x,
        "y_src": y,
        "test_size": test_size,
        "x_test": x_test,
        "y_test": y_test,
        "colors": ["red", "blue"],
    }
    visualise_results(**input_map)

    if auto_grid:
        print("\nBest parameters: {}".format(grid.best_params_))


if __name__ == "__main__":
    main(C=10, gamma=0.9, test_size=0.3, auto_grid=True)
