import pandas as pd
import torch
from torch.utils.data import Dataset

# Add csv loaderclass
class SpiralDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None).values
        self.x = torch.tensor(self.data[:, :2], dtype=torch.float32)
        self.y = torch.tensor(self.data[:, 2], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    # Fill this member so we can get data by index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
