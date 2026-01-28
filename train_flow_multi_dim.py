import time

from torch.utils.data import DataLoader, random_split

from nflows.transforms.autoregressive import *

import matplotlib.pyplot as plt
import pandas as pd

from data import EventDataset
from utils.loss import (
    calculate_impossible_mass_penalty,
    calculate_first_order_non_smoothness_penalty,
    calculate_outlier_gradient_penalty_with_preprocess_mod_z_scores,
)
from models.flows import create_spline_flow
from settings import TEST_PROPORTION, RANDOM_SEED
from utils.physics import calculate_dijet_mass


# Set training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
EPOCHS = 5
DATASET_SIZE = 500_000

SMOOTHNESS_PENALTY_FACTOR = 0.1
IMPOSSIBLE_MASS_PENALTY_FACTOR = 0.0

# Load settings from pre-processing
settings = pd.read_csv("pre_process_results/multi_dim_50_epochs_s0_b4096_p001.csv")
GRAD_MEDIAN = settings["first_order_median"].item()
GRAD_MAD = settings["first_order_mad"].item()

# Prepare dataset
data = EventDataset(
    "data/background.csv",
    "data/signal.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    DATASET_SIZE,
    signal_proportion=0.1,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="multi_dim",
)
train_data, test_data = random_split(
    data,
    [1 - TEST_PROPORTION, TEST_PROPORTION],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)
train_dataloader = DataLoader(train_data, BATCH_SIZE)
test_dataloader = DataLoader(test_data, BATCH_SIZE)

# Load and prepare model
flow = create_spline_flow(10, 8, 32, 64, 4.0)

# Set up training optimizer
optimizer = torch.optim.AdamW(
    flow.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Establish performance metrics
loss_per_epoch = []

# Train the flow
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    for batch, (X, sb_truth) in enumerate(
        train_dataloader
    ):  # sb_truth = actual s/b label (1/0 respectively)
        optimizer.zero_grad()

        # Calculate the log probability and the log absolute determinant
        z, logabsdet = flow._transform.forward(X)
        log_prob_z = flow._distribution.log_prob(z)
        log_prob = log_prob_z + logabsdet

        # Calculate negative log likelihood for distribution
        loss = -log_prob.mean()

        # Calculate un-smoothness penalty
        if SMOOTHNESS_PENALTY_FACTOR > 0.0:
            penalty = calculate_outlier_gradient_penalty_with_preprocess_mod_z_scores(
                log_prob,
                X,
                GRAD_MEDIAN,
                GRAD_MAD,
                1.0,
                SMOOTHNESS_PENALTY_FACTOR,
                -1.0,
            )
            loss += penalty

        # Calculate impossible mass penalty
        if IMPOSSIBLE_MASS_PENALTY_FACTOR > 0.0:
            mass_penalty = calculate_impossible_mass_penalty(
                flow, 1000, IMPOSSIBLE_MASS_PENALTY_FACTOR
            )
            loss += mass_penalty

        loss.backward()
        optimizer.step()

        # Display training metrics
        if batch % 100 == 0:
            print(f"{batch} - loss: {loss}")

    with torch.no_grad():
        test_loss = 0
        for X, y in test_dataloader:
            loss = -flow.log_prob(X).mean()
            test_loss += loss / len(test_dataloader)

        loss_per_epoch.append(test_loss)
        print(f"Testing loss: {test_loss}")

# Save model
model_save_name = input("\nSaved model file name [time/date]: ")
if model_save_name.strip() == "":
    model_save_name = round(time.time())
torch.save(flow.state_dict(), f"saved_models_multi_dim/{model_save_name}.pth")

# Plot the loss over time
plt.plot(loss_per_epoch, color="tab:blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Test the flow's generation
with torch.no_grad():
    bins = 50
    limit = [0, 1]

    fake_events = calculate_dijet_mass(flow.sample(DATASET_SIZE))

    figure, axis = plt.subplots(1, 2, sharex=True, sharey=True)
    axis[0].hist(
        [calculate_dijet_mass(data.features).flatten()],
        bins=bins,
        histtype="bar",
        color="black",
        label="Original",
        range=limit,
    )
    axis[1].hist(
        [fake_events.flatten()],
        bins=bins,
        histtype="bar",
        color="tab:red",
        label="Generated",
        range=limit,
    )
    figure.supxlabel("Mass")
    figure.supylabel("Entries")
    figure.legend()
    plt.show()
