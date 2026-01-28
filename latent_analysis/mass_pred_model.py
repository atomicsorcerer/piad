import torch
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib
from torch.utils.data import random_split, DataLoader

from LatentDataset import LatentDataset
from settings import RANDOM_SEED

# Set training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.01
EPOCHS = 10
TEST_PROPORTION = 0.0

dataset = LatentDataset(
    "../data/latent/latent_0.csv",
    ["normalized_predicted_density", "normalized_gradient_norm"],
    "normalized_mass",
)
train_data, test_data = random_split(
    dataset,
    [1 - TEST_PROPORTION, TEST_PROPORTION],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)
train_dataloader = DataLoader(train_data, BATCH_SIZE)
test_dataloader = DataLoader(test_data, BATCH_SIZE)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Linear(64, 1),
    torch.nn.BatchNorm1d(1),
    torch.nn.Sigmoid(),
)

# Set up training optimizer and loss function
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
loss_function = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    for batch, (X, Y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_function(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>5f}")

model_save_name = input("Saved model file name [time/date]: ")
torch.save(model.state_dict(), f"saved/{model_save_name}.pth")
