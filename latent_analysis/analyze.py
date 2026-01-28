import torch
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib
from torch.utils.data import random_split, DataLoader

from LatentDataset import LatentDataset
from settings import RANDOM_SEED


plot_set = LatentDataset(
    "../data/latent/latent_0.csv",
    ["normalized_predicted_density", "normalized_gradient_norm"],
    "normalized_mass",
)

# plt.hist2d(
#     plot_set.features[..., 0][plot_set.sb_truth == 1.0],
#     plot_set.features[..., 1][plot_set.sb_truth == 1.0],
#     bins=100,
#     cmap="plasma",
#     # range=((-8, -8), (8, 8)),
# )
# plt.show()
#
# # plt.scatter(plot_set.features[..., 0], plot_set.features[..., 1], s=0.01)
# plt.show()
# exit()

# model = torch.nn.Sequential(
#     torch.nn.Linear(2, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 64),
#     torch.nn.ReLU(),
#     torch.nn.Linear(64, 1),
#     torch.nn.Sigmoid(),
# )
# model.load_state_dict(torch.load("saved/latent_1.pth"))
#
# pred_masses = model(plot_set.features).flatten().detach()
#
# plt.hist(pred_masses)
# plt.show()
# exit()

sorted_indices = torch.argsort(plot_set.labels.flatten())
sorted_features = plot_set.features[sorted_indices]
sorted_coords = sorted_features[..., :2]
sorted_masses = plot_set.labels[sorted_indices].flatten()

SEARCH_CHUNKS = 50
search_radius = 0.5 / SEARCH_CHUNKS
# search_radius = 0.005
search_space = torch.linspace(min(sorted_masses), max(sorted_masses), SEARCH_CHUNKS + 1)

result_array = torch.zeros(len(search_space))
for i, mass in enumerate(search_space):
    valid_points = sorted_coords[
        (mass - search_radius <= sorted_masses)
        & (sorted_masses <= mass + search_radius)
    ]
    if len(valid_points) == 0:
        continue

    valid_points_repeated = valid_points.repeat(len(valid_points), 1).reshape(
        len(valid_points), len(valid_points), 2
    )
    diffs = valid_points_repeated - valid_points.reshape(len(valid_points), 1, 2)
    dists = torch.norm(diffs, dim=1)
    result_array[i] = torch.mean(dists).item()

# Graph it
fig, ax1 = plt.subplots()
ax1.set_ylabel("Mean of distances", color="tab:red")
ax1.set_xlabel("Normalized mass")
ax1.plot(search_space, result_array, color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Signal events density", color="black")
ax2.hist(
    plot_set.labels.flatten()[plot_set.sb_truth == 1.0],
    density=True,
    color="black",
    histtype="step",
    alpha=0.5,
    bins=50,
)
ax2.tick_params(axis="y", labelcolor="black")

fig.tight_layout()
plt.show()
exit()

slice_size = 5
result_array = torch.zeros(len(sorted_indices) - slice_size + 1)
for i in range(len(sorted_indices) - slice_size + 1):
    coords = sorted_coords[i : (i + slice_size)]
    diffs = coords[0] - coords
    dists = torch.norm(diffs, dim=1)
    result_array[i] = torch.sum(dists).item()

plt.scatter(sorted_features[..., 2][: -slice_size + 1], result_array, s=0.1)
plt.show()
exit()


# X = plot_set.features[plot_set.sb_truth == 1.0][..., 0]
# Y = plot_set.features[plot_set.sb_truth == 1.0][..., 1]
# Z = plot_set.features[plot_set.sb_truth == 1.0][..., 2]
# # cmap = matplotlib.colormaps["coolwarm"]
# # color = cmap(Z)[..., :3]
# # plt.scatter(X, Y, c=color, s=1, alpha=0.5)
# # plt.show()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_trisurf(
#     X,
#     Y,
#     Z,
#     cmap=plt.cm.coolwarm,
#     linewidth=0,
#     antialiased=False,
# )
# plt.setp(ax, xlim=(-4, 10), ylim=(-8, 2), zlim=(0, 1))
# plt.show()
#
# exit()
