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
    "../data/background.csv",
    "../data/signal_files/pp-y1-jj_1000GeV.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    100_000,
    0.01,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="multi_dim",
)
X = data.features
Y = data.labels.flatten()

unconstrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
unconstrained_state_dict = torch.load(
    "../saved_models/pp-y1-jj_1000GeV/multi_dim_100_epochs_s00_b16384_p001_0.pth"
)
unconstrained_state_dict = convert_cluster_state_dict(unconstrained_state_dict)
unconstrained_flow.load_state_dict(unconstrained_state_dict)
unconstrained_Y = unconstrained_flow.log_prob(X)

unconstrained_Y_median = unconstrained_Y.median()
unconstrained_Y_mad = abs(unconstrained_Y - unconstrained_Y_median).median()
unconstrained_Y_modified_z_score = (
    0.6745 * (unconstrained_Y - unconstrained_Y_median) / unconstrained_Y_mad
)

grad_log_prob_first = torch.autograd.grad(
    outputs=unconstrained_Y.sum(), inputs=X, create_graph=True
)[0]
gradients_first_order = torch.norm(grad_log_prob_first, dim=1)
first_order_median = gradients_first_order.median()
first_order_mad = abs(gradients_first_order - first_order_median).median()
modified_z_score_first_order = (
    0.6745 * (gradients_first_order - first_order_median) / first_order_mad
)

# plt.scatter(
#     unconstrained_Y_modified_z_score.detach().numpy(),
#     modified_z_score_first_order.detach().numpy(),
# )
# plt.hist2d(
#     unconstrained_Y_modified_z_score.detach().numpy()[Y == 1.0],
#     modified_z_score_first_order.detach().numpy()[Y == 1.0],
#     bins=50,
#     cmap="plasma",
#     # range=((-8, -8), (8, 8)),
# )
# plt.show()
# exit()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 8), sharex=True, sharey=True)
axes = axes.flatten()

axes[0].hist2d(
    unconstrained_Y_modified_z_score.detach().numpy(),
    modified_z_score_first_order.detach().numpy(),
    bins=100,
    cmap="plasma",
)
axes[0].set_title("Full dataset")
axes[1].hist2d(
    unconstrained_Y_modified_z_score.detach().numpy()[Y == 0.0],
    modified_z_score_first_order.detach().numpy()[Y == 0.0],
    bins=100,
    cmap="plasma",
)
axes[1].set_title("Background only")
axes[2].hist2d(
    unconstrained_Y_modified_z_score.detach().numpy()[Y == 1.0],
    modified_z_score_first_order.detach().numpy()[Y == 1.0],
    bins=100,
    cmap="plasma",
)
axes[2].set_title("Signal only")
# plt.setp(axes, xlim=(-4, 4), ylim=(-4, 4))
fig.supxlabel("Density prediction (modified z-score)")
fig.supylabel("Gradient norm (modified z-score)")
plt.tight_layout()
plt.show()

output = {
    "gradient_norm": gradients_first_order.flatten().tolist(),
    "normalized_gradient_norm": modified_z_score_first_order.flatten().tolist(),
    "predicted_density": unconstrained_Y.flatten().tolist(),
    "normalized_predicted_density": unconstrained_Y_modified_z_score.flatten().tolist(),
    "mass": data.mass.flatten().tolist(),
    "normalized_mass": (data.mass / data.mass.max()).flatten().tolist(),
    "labels": data.labels.flatten().tolist(),
}
settings = pl.DataFrame(output)
save_name = input("Save output as: ")
if save_name.strip() != "":
    settings.write_csv(f"../data/latent/{save_name}.csv")
