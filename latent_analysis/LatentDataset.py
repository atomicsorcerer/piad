import pandas as pd
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(self, db_source: str, features: list[str], labels="mass") -> None:
        dataset = pd.read_csv(db_source)
        labels = dataset[labels]
        features = dataset[features]
        sb_labels = dataset["labels"]

        self.labels = (
            torch.from_numpy(labels.values).type(torch.float32).reshape((-1, 1))
        )
        self.features = torch.from_numpy(features.values).type(torch.float32)
        self.sb_truth = torch.from_numpy(sb_labels.values).type(torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
