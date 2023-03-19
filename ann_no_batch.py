import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


from dataset.spiral_ds_loader import SpiralDataset
from networks.ann import ANN


# This script is here to mainly show what happens to the loss and accuracy plots when we only iterate through the training set once
# instead of training by mini batches
# Compute final prediction for test set
def test(model, x_test, y_test):
    y_test = y_test.to("cpu")

    with torch.no_grad():
        y_pred_probs_test = model(x_test).cpu().numpy()

    # Convert prediction probabilities to classes with cutoff 0.5
    y_pred_classes_test = np.argmax(y_pred_probs_test, axis=1)

    # Get accuracy, precision and recall
    confusion_mat = confusion_matrix(y_test, y_pred_classes_test)

    acc = accuracy_score(y_test, y_pred_classes_test)

    precision_global = precision_score(y_test, y_pred_classes_test, average="micro")
    precision_mean = precision_score(y_test, y_pred_classes_test, average="macro")

    recall_global = recall_score(y_test, y_pred_classes_test, average="micro")
    recall_mean = recall_score(y_test, y_pred_classes_test, average="macro")

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
    # Generate meshgrid
    x_min, x_max, y_min, y_max = extents
    x, y = np.meshgrid(
        np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points)
    )

    # Parse grid to input that the model can process
    grid = np.stack((x.ravel(), y.ravel()), axis=1)

    # Generate predictions
    with torch.no_grad():
        # Make sure everything is on CPU
        model = model.to("cpu")
        model.eval()
        inputs = torch.tensor(grid, dtype=torch.float32)
        predictions = model(inputs).numpy()
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


def main(
    data_path,
    lr,
    num_epochs,
    batch_size,
    input_layers,
    hidden_layers,
    output_layers,
    activation,
    criterion,
):
    # Set device - GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using {device} device")

    # Create dataset
    dataset = SpiralDataset(data_path)
    x = dataset.x
    y = dataset.y

    # Show dataset
    # plt.scatter(dataset.data[:, 0], dataset.data[:, 1])
    # plt.show()

    # Split dataset into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.3, random_state=123, shuffle=True
    )

    # Get test set from validation set
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, test_size=0.5, random_state=123, shuffle=False
    )

    # Create model, optimiser, and loss function
    model = ANN(
        input_layers,
        hidden_layers,
        output_layers,
        activation,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # History logging
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # Train model
    beginning = time.time()
    for epoch in range(num_epochs):
        # Reset gradients
        optimiser.zero_grad()

        y_pred = model(x_train)

        # Calculate loss
        loss = criterion(y_pred, y_train)

        # Backpropagate then update weights
        loss.backward()
        optimiser.step()

        # Calculate metrics
        train_acc = torch.sum(torch.argmax(y_pred, dim=1) == y_train) / y_train.shape[0]
        train_losses.append(loss.item())
        train_accs.append(train_acc)

        # Calculate validation
        with torch.no_grad():
            y_pred = model(x_val)
            loss_val = criterion(y_pred, y_val)
            val_acc = torch.sum(torch.argmax(y_pred, dim=1) == y_val) / y_val.shape[0]
            val_losses.append(loss_val.item())
            val_accs.append(val_acc)

        if epoch % 10 == 0:
            # Print to training and validation metrics to console
            print(
                "Epoch {}:\tTrain loss={:.4f}  \tTrain acc={:.2f} \tVal loss={:.4f} \tVal acc={:.2f}".format(
                    epoch, loss.item(), train_acc * 100, loss_val.item(), val_acc * 100
                )
            )
    end = time.time()

    torch.save(model.state_dict(), "spiral_model.pt")

    print(
        "\nTraining finished! Time elapsed: {:.4f} minutes\n".format(
            (end - beginning) / 60.0
        )
    )

    # Print confusion matrix, accuracy, precision and recall
    metrics_map = test(model, x_test, y_test)

    print("Confusion matrix: \n{}\n".format(metrics_map["conf_mat"]))
    print(" Accuracy - {:.4f}".format(metrics_map["acc"]))
    print(
        "Precision - Global: {:.4f} \t Mean: {:.4f}".format(
            metrics_map["precision_global"], metrics_map["precision_mean"]
        )
    )
    print(
        "   Recall - Global: {:.4f} \t Mean: {:.4f}".format(
            metrics_map["recall_global"], metrics_map["recall_mean"]
        )
    )

    # Visualise results
    logs = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }

    print("\nVisualising results...")
    visualise_results(model, logs, extents=[-6, 6, -6, 6], num_points=1000)


if __name__ == "__main__":

    # Create input map and then unpack to call main
    i_args = {
        "data_path": "dataset/spiralsdataset.csv",
        "lr": 0.01,
        "num_epochs": 1000,
        "batch_size": 10,
        "input_layers": 2,
        "hidden_layers": [140, 140],
        "output_layers": 2,
        "activation": torch.nn.ReLU(),
        "criterion": torch.nn.CrossEntropyLoss(),
    }
    main(**i_args)
