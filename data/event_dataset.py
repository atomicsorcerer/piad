from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from settings import RANDOM_SEED


class EventDataset(Dataset):
    def __init__(
        self,
        bg_file_path: str,
        signal_file_path: str,
        included_features: list[str],
        limit: int = 10_000,
        signal_proportion: float = 0.5,
        normalize: bool = False,
        norm_type: Literal["one_dim", "multi_dim"] = "1d",
        mass_region: tuple[float, float | None] | None = None,
    ) -> None:
        # Import the CSV files and add a column for background/signal (denoted as 0 or 1, respectively)
        bg_dataset = pd.read_csv(bg_file_path).assign(label=0.0)
        signal_dataset = pd.read_csv(signal_file_path).assign(label=1.0)

        # Remove events that are outside of the physical mass region
        if mass_region is not None:
            bg_dataset = bg_dataset.loc[bg_dataset["mass"] >= mass_region[0]]
            signal_dataset = signal_dataset.loc[
                signal_dataset["mass"] >= mass_region[0]
            ]

        if mass_region[1] is not None:
            bg_dataset = bg_dataset.loc[bg_dataset["mass"] <= mass_region[1]]
            signal_dataset = signal_dataset.loc[
                signal_dataset["mass"] <= mass_region[1]
            ]

        # Sample the dataset
        if (limit * signal_proportion) % 1 != 0:
            raise ValueError("Limit times the signal proportion must be an integer.")

        signal_dataset = signal_dataset.sample(
            n=int(signal_proportion * limit),
            random_state=RANDOM_SEED,
        )
        bg_dataset = bg_dataset.sample(
            n=int((1.0 - signal_proportion) * limit),
            random_state=RANDOM_SEED,
        )

        dataset = pd.concat((bg_dataset, signal_dataset))

        # Select the necessary columns for training, split dataset into features and labels
        features = dataset[included_features]
        labels = dataset["label"]

        # Convert dataset type to torch.Tensor and reshape it
        features = torch.from_numpy(features.values).type(torch.float32)
        labels = torch.from_numpy(labels.values).type(torch.float32).reshape((-1, 1))

        # Potential normalization of the dataset (may remove events on the extremes)
        if normalize:
            if norm_type == "one_dim":
                # Based on the mass normalization techniques used for CATHODE (https://arxiv.org/pdf/2109.00546)
                features = (features - features.min()) / (
                    features.max() - features.min()
                )
                labels = labels[(features != 0) & (features != 1)]
                features = features[(features != 0) & (features != 1)]
                features = torch.log(features / (1 - features))
                features = features - features.mean()
                features = features / features.std()
            if norm_type == "multi_dim":
                self.normalizing_factor = features.abs().max()
                features = features / self.normalizing_factor

        features = features.reshape((-1, len(included_features)))
        features.requires_grad_()

        self.mass = (
            torch.from_numpy(dataset["mass"].values)
            .type(torch.float32)
            .reshape((-1, 1))
        )
        self.dataframe = dataset

        # Move data to GPU, if applicable
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.features = features.to(device)
            self.labels = labels.to(device)
            print(f"Data moved to {torch.cuda.get_device_name(0)}.")
        else:
            self.features = features
            self.labels = labels
            print(f"Data on CPU.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ParameterizedBackgroundDataset(Dataset):
    def __init__(
        self,
        start: int,
        stop: int,
        bins: int,
        params: list[tuple[float, float, float, float]],
        noise_scale: float = 1.0,
    ) -> None:
        self.start = start
        self.stop = stop
        self.bins = bins
        self.params = params

        step = (stop - start) / bins
        params_dataset = []
        mass_dataset = []
        for theta_0, theta_1, theta_2, theta_3 in params:
            f = (
                lambda x: theta_0
                * (1 - x) ** theta_1
                * x**theta_2
                * x ** (theta_3 * np.log(x))
            )  # General form of a smooth mass background distribution

            for i in range(bins):
                val = start + (i + 0.5) * step  # Middle of the bin interval
                n_in_bin = round(f(val))
                params_dataset.extend(
                    [theta_0, theta_1, theta_2, theta_3] * n_in_bin
                )  # Append params for each in the smooth distribution function
                mass_dataset.extend(
                    [val] * n_in_bin
                )  # Append 'val' for each in the smooth distribution function

        mass_dataset = torch.Tensor(mass_dataset).reshape(-1, 1)
        entry_noise = (
            (
                torch.rand(
                    mass_dataset.shape,
                    generator=torch.Generator().manual_seed(RANDOM_SEED),
                )
                - 0.5
            )
            * step
            * noise_scale
        )
        self.mass_dataset = mass_dataset + entry_noise
        self.params_dataset = torch.Tensor(params_dataset).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.mass_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.params_dataset[idx], self.mass_dataset[idx]


if __name__ == "__main__":
    data = EventDataset(
        "background.csv",
        "signal_files/pp-y1-jj_1250GeV.csv",
        ["mass"],
        100_000,
        mass_region=(500.0, None),
        signal_proportion=0.1,
        normalize=False,
        norm_type="one_dim",
    )
    signal = (
        data.features.detach().numpy()[data.labels.flatten().detach() == 1.0].flatten()
    )
    bg = data.features.detach().numpy()[data.labels.flatten().detach() == 0.0].flatten()
    plt.hist(
        [bg, signal],
        bins=300,
        histtype="barstacked",
        label=["Background", "Signal"],
        # range=(500, 3000),
    )
    plt.xlabel("Mass")
    plt.ylabel("Entries")
    plt.legend()
    plt.title("Signal proportion = 1%")
    plt.show()
