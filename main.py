import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from torch.utils.data import DataLoader

from dataset.spiral_ds_loader import SpiralDataset
from networks.ann import ANN

# TODO Get test set, implement cross validation
# TODO implement precision, recall, f1 score

# Add training function
def train(model, train_loader, criterion, optimiser, device):
    # Let model know we are in training mode
    model.train()

    # Keep track of training loss and accuracy
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        # Cast tensors to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Reset gradients
        optimiser.zero_grad()

        # Get model outputs and calculate loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagate and update optimiser learning rate
        loss.backward()
        optimiser.step()

        # Keep track of loss and accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = train_loss / len(train_loader)

    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    # Let model know we are in evaluation mode
    model.eval()

    # Keep track of validation loss
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Cast tensors to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Calculate model output and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Keep track of loss and accuracy
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = val_loss / len(val_loader)

    return avg_loss, acc


# Compute final prediction for test set
def test(model, x_test, y_test):
    y_test = y_test.to("cpu")

    with torch.no_grad():
        y_pred_probs_test = model(x_test).cpu().numpy()

    # Convert prediction probabilities to classes with cutoff 0.5
    y_pred_classes_test = np.argmax(y_pred_probs_test, axis=1)
    confusion_mat = confusion_matrix(y_test, y_pred_classes_test)

    # Get accuracy, precision and recall
    acc = accuracy_score(y_test, y_pred_classes_test)
    precision = precision_score(y_test, y_pred_classes_test)
    recall = recall_score(y_test, y_pred_classes_test)

    return confusion_mat, acc, precision, recall


def main(
    data_path,
    lr,
    num_epochs,
    batch_size,
    input_layers,
    hidden_layers,
    hidden_size,
    output_layers,
    activation,
    criterion,
):
    # Set device - GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Create dataset
    dataset = SpiralDataset(data_path)

    # Show dataset
    # plt.scatter(dataset.data[:, 0], dataset.data[:, 1])
    # plt.show()

    # Split dataset into train, validation, and test sets
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=0.3, random_state=42, shuffle=True
    )

    val_dataset, test_dataset = train_test_split(
        val_dataset, test_size=0.5, random_state=42, shuffle=False
    )

    test_dataset = torch.tensor(
        [[x[0], x[1], _] for x, _ in test_dataset], dtype=torch.float32, device=device
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model, optimiser, and loss function
    model = ANN(
        input_layers,
        np.array(hidden_size * np.ones(hidden_layers), dtype=np.int32),
        output_layers,
        activation,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # History logging
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # Train model
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimiser, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
                f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%"
            )

        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc

    torch.save(model.state_dict(), "spiral_model.pt")

    print("\nTraining finished!\n")

    # Print confusion matrix, accuracy, precision and recall
    confusion_mat, acc, precision, recall = test(
        model, test_dataset[..., :2], test_dataset[..., 2]
    )

    print(f"Confusion matrix: \n{confusion_mat}\n")
    print(
        f"Accuracy: {acc*100:.4f}% \tPrecision: {precision:.4f} \tRecall: {recall:.4f}"
    )

    # Plot training and validation history
    plt.figure()
    plt.title("Training vs Validation History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses, "r")
    plt.plot(val_losses, "g")
    plt.legend(["Train loss", "Validation loss"])

    plt.figure()
    plt.title("Training vs Validation History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_accs, "b")
    plt.plot(val_accs, "y")
    plt.legend(["Train accuracy", "Validation accuracy"])

    plt.show()


# Function for visualising model
def visualize_spiral(model, extents, num_points):
    # Generate your meshgrid
    x_min, x_max, y_min, y_max = extents
    x, y = np.meshgrid(
        np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points)
    )
    grid = np.stack((x.ravel(), y.ravel()), axis=1)

    # Generate your predictions
    with torch.no_grad():
        model.eval()
        inputs = torch.tensor(grid, dtype=torch.float32)
        predictions = model(inputs).numpy()
        predictions = np.reshape(predictions, (num_points, num_points, 2))

    predictions = np.stack(
        (predictions[..., 0], predictions[..., 1], np.zeros(predictions[..., 0].shape)),
        axis=-1,
    )

    labels = np.argmax(predictions[..., :2], axis=2)

    plt.title("Pixel map")
    plt.scatter(grid[:, 0], grid[:, 1], s=1, c=labels.ravel(), cmap="coolwarm")
    plt.show()


if __name__ == "__main__":

    # Create input map and then unpack to call main
    i_args = {
        "data_path": "dataset/spiralsdataset.csv",
        "lr": 0.01,
        "num_epochs": 1000,
        "batch_size": 10,
        "input_layers": 2,
        "hidden_layers": 1,
        "hidden_size": 64,
        "output_layers": 2,
        "activation": torch.nn.Tanh(),
        "criterion": torch.nn.CrossEntropyLoss(),
    }
    main(**i_args)

    # Visualise model if desired
    visualise = True
    if visualise:
        model = ANN(
            i_args["input_layers"],
            np.array(
                i_args["hidden_size"] * np.ones(i_args["hidden_layers"]), dtype=np.int32
            ),
            i_args["output_layers"],
            i_args["activation"],
        )
        model.load_state_dict(torch.load("spiral_model.pt"))

        visualize_spiral(model, extents=[-10, 10, -10, 10], num_points=1000)
