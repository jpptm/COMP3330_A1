import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from networks.ann import SpiralNet


class SpiralDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None).values
        self.x = torch.tensor(self.data[:, :2], dtype=torch.float32)
        self.y = torch.tensor(self.data[:, 2], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# dataset = SpiralDataset("dataset/spiralsdataset.csv").__getitem__(0)
# print(dataset)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = train_loss / len(train_loader)
    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = val_loss / len(val_loader)
    return avg_loss, acc


def loss_func():
    def rmse(ytrue, ypred):
        return torch.sqrt(torch.mean((ytrue - ypred) ** 2))

    return rmse


def main():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using {device} device")

    # set hyperparameters
    lr = 0.01
    num_epochs = 20000
    batch_size = 16

    # create dataset
    dataset = SpiralDataset("dataset\spiralsdataset.csv")

    # split dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=0.3, random_state=42
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # create model, optimizer, and loss function
    model = SpiralNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # train and validate model
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
                f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%"
            )

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "spiral_model.pt")

    print("Training finished!")


# Test visualisation
def visualise(test_data, model_path):
    # Load model
    model = SpiralNet()
    model.load_state_dict(model_path)

    # Set model to evaluation mode
    model.eval()

    # Create data loader for test data

    test_dataset = SpiralDataset(test_data)
    _, vd = train_test_split(test_dataset, test_size=0.5, random_state=123)
    vd_loader = DataLoader(vd, batch_size=1)

    # Iterate over test data and compute model predictions
    plt.grid("on")
    plt.legend()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(vd_loader):
            # Forward pass
            outputs = model(inputs.float())
            # Convert outputs and targets to numpy arrays
            outputs_np = np.argmax(np.squeeze(outputs.numpy()))
            targets_np = targets.numpy()
            # print(test_dataset)
            # print(outputs_np)
            # print(targets_np)
            # print(test_dataset)
            # Plot the spiral dataset with model predictions and ground truth values
            item = test_dataset.__getitem__(idx)
            # print(item)
            plt.scatter(item[0][0], item[0][1], c=item[1])
            plt.scatter(inputs[0][0], inputs[0][1], c="green", label="Input")
            plt.scatter(inputs[0][0], inputs[0][1], c=outputs_np, label="Prediction")
            plt.scatter(inputs[0][0], inputs[0][1], c=targets_np, label="Ground Truth")
        #
        plt.show()


if __name__ == "__main__":
    # main()
    test_data = np.stack(
        np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1)), -1
    )
    visualise("dataset\spiralsdataset.csv", torch.load("spiral_model.pt"))
