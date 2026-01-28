import numpy as np
import torch
from matplotlib import pyplot as plt
import polars as pl

from data import EventDataset
from models.flows import create_spline_flow


def convert_cluster_state_dict(state_dict):
    # Create a new state_dict without the "module." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    return new_state_dict


bins = 100
z_bins = 30

data = EventDataset(
    "data/background.csv",
    "data/signal.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    10_000,
    0.01,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="multi_dim",
)
X = data.features
Y = data.labels.flatten()

unconstrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
unconstrained_state_dict = torch.load(
    "saved_models/fixed_loss_multi_dim_100_epochs_s00_b16384_p001_3.pth"
)
unconstrained_state_dict = convert_cluster_state_dict(unconstrained_state_dict)
unconstrained_flow.load_state_dict(unconstrained_state_dict)
unconstrained_Y = unconstrained_flow.log_prob(X)

grad_log_prob_first = torch.autograd.grad(
    outputs=unconstrained_Y.sum(), inputs=X, create_graph=True
)[0]
grad_log_prob_second = torch.autograd.grad(
    outputs=grad_log_prob_first.sum(), inputs=X, create_graph=True
)[0]

gradients_first_order = torch.norm(grad_log_prob_first, dim=1)
grad_std_dev_first_order = torch.std(gradients_first_order)
grad_mean_first_order = torch.mean(gradients_first_order)
z_scores_first_order = (
    gradients_first_order - grad_mean_first_order
) / grad_std_dev_first_order

first_order_median = gradients_first_order.median()
first_order_mad = abs(gradients_first_order - first_order_median).median()
modified_z_score_first_order = (
    0.6745 * (gradients_first_order - first_order_median) / first_order_mad
)

# gradients_second_order = torch.norm(grad_log_prob_second, dim=1)
# grad_std_dev_second_order = torch.std(gradients_second_order)
# grad_mean_second_order = torch.mean(gradients_second_order)
# z_scores_second_order = (
#     gradients_second_order - grad_mean_second_order
# ) / grad_std_dev_second_order
#
# second_order_median = gradients_second_order.median()
# second_order_mad = abs(gradients_second_order - second_order_median).median()
# modified_z_score_second_order = (
#     0.6745 * (gradients_second_order - second_order_median) / second_order_mad
# )

labels = [
    "First jet energy",
    "First jet p_x",
    "First jet p_y",
    "First jet p_z",
    "Second jet energy",
    "Second jet p_x",
    "Second jet p_y",
    "Second jet p_z",
]
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, axis in enumerate(axes):
    if i >= len(labels):
        break

    ind_grads = grad_log_prob_first[..., i].abs()
    axis.hist(
        [
            ind_grads[Y == 0.0].detach().numpy(),
            ind_grads[Y == 1.0].detach().numpy(),
        ],
        range=(0, 1000),
        bins=z_bins,
        color=["tab:blue", "tab:red"],
        # label=["Background", "Signal"],
        histtype="step",
        density=True,
    )
    axis.set_title(labels[i])

fig.legend(["Signal", "Background"], loc="upper right", bbox_to_anchor=(0.97, 0.95))
fig.supxlabel("First-order gradient magnitude of given input dimension")
fig.supylabel("Density")
plt.tight_layout()
plt.show()
