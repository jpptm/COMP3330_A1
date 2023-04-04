import pandas as pd
import torch
from torch.utils.data import Dataset

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
            # x1x2 = self.x[:, 0] * self.x[:, 1]

            self.x = torch.stack([self.x[:, 0], self.x[:, 1], r, theta], dim=-1)

    def __len__(self):
        return len(self.data)

    # Fill this member so we can get data by index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
