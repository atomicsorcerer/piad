import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from torcheval.metrics import BinaryAUROC

from matplotlib import pyplot as plt
import polars as pl

from pfn_utils import train, test
from model import ParticleFlowNetwork
from data.event_dataset import EventDataset
from settings import TEST_PROPORTION, RANDOM_SEED

BATCH_SIZE = 64
data = EventDataset(
    "../data/background.csv",
    "../data/signal.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    100_000,
    signal_proportion=0.5,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="multi_dim",
)
data.features = data.features.reshape((-1, 2, 4))

batch_size = 128

train_data, test_data = random_split(
    data,
    [1 - TEST_PROPORTION, TEST_PROPORTION],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)
train_dataloader = DataLoader(train_data, BATCH_SIZE)
test_dataloader = DataLoader(test_data, BATCH_SIZE)

model = ParticleFlowNetwork(4, 8, 16, [512, 256, 128], [128, 128])

lr = 0.00001
weight_decay = 0.01
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)

epochs = 100
loss_over_time = []
accuracy_over_time = []
auc = []
max_acc = 0.0
max_acc_epoch = 0
metric = BinaryAUROC()
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_function, optimizer, True)
    loss, acc, auc_metric = test(test_dataloader, model, loss_function, metric, True)

    loss_over_time.append(loss)
    accuracy_over_time.append(acc)
    auc.append(auc_metric)

    if acc > max_acc:
        max_acc = acc
        max_acc_epoch = t + 1

print("Finished Training")

torch.save(model, "model.pth")
print("Saved Model")

print(
    f"Model saved had {max_acc * 100:<0.2f}% accuracy, and was from epoch {max_acc_epoch}."
)

plt.plot(accuracy_over_time[0:max_acc_epoch])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("PFN Classifier Accuracy per Epoch")
plt.show()
