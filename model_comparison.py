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


bins = 200
z_bins = 50

data = EventDataset(
    "data/background.csv",
    "data/signal.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    10_000,
    0.1,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="multi_dim",
)
X = data.features
Y = data.labels.flatten()

flow_0_state_dict = torch.load(
    "saved_models_multi_dim_DEP/multi_dim_50_epochs_s0_b16384_p001.pth"
)
flow_1_state_dict = torch.load(
    "saved_models_multi_dim_DEP/multi_dim_50_epochs_s01_b16384_p001.pth"
)
# flow_2_state_dict = torch.load(
#     "saved_models_multi_dim_DEP/multi_dim_50_epochs_s0_b4096.pth"
# )

flow_0 = create_spline_flow(10, 8, 32, 64, 4.0)
flow_0_state_dict = convert_cluster_state_dict(flow_0_state_dict)
flow_0.load_state_dict(flow_0_state_dict)
flow_0_Y = flow_0.log_prob(X)

flow_1 = create_spline_flow(10, 8, 32, 64, 4.0)
flow_1_state_dict = convert_cluster_state_dict(flow_1_state_dict)
flow_1.load_state_dict(flow_1_state_dict)
flow_1_Y = flow_1.log_prob(X)

# flow_2 = create_spline_flow(10, 8, 32, 64, 4.0)
# flow_2_state_dict = convert_cluster_state_dict(flow_2_state_dict)
# flow_2.load_state_dict(flow_2_state_dict)
# flow_2_Y = flow_2.log_prob(X)

grad_log_prob_first_0 = torch.autograd.grad(
    outputs=flow_0_Y.sum(), inputs=X, create_graph=True
)[0]
grad_log_prob_first_1 = torch.autograd.grad(
    outputs=flow_1_Y.sum(), inputs=X, create_graph=True
)[0]
# grad_log_prob_first_2 = torch.autograd.grad(
#     outputs=flow_2_Y.sum(), inputs=X, create_graph=True
# )[0]

model_0_gradients_first_order = torch.norm(grad_log_prob_first_0, dim=1)
model_1_gradients_first_order = torch.norm(grad_log_prob_first_1, dim=1)
# model_2_gradients_first_order = torch.norm(grad_log_prob_first_2, dim=1)

# Modified z-score histogram
plt.hist(
    [
        model_0_gradients_first_order.detach().numpy(),
        model_1_gradients_first_order.detach().numpy(),
        # model_2_gradients_first_order.detach().numpy(),
    ],
    bins=z_bins,
    color=["black", "tab:red"],
    label=[
        "Regularization alpha = 0.0",
        "Regularization alpha = 0.1",
        # "Global batch size: 4096",
    ],
    # histtype="barstacked",
)
plt.xlabel("Gradient magnitude")
plt.ylabel("Entries")
plt.legend()
plt.title("Unconstrained, trained for 50 epochs with same other hyperparams")
plt.show()
