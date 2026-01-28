import numpy as np
import torch
from click import style
from matplotlib import pyplot as plt
from sklearn import metrics

from data import EventDataset
from models.flows import create_spline_flow
from utils.physics import calculate_dijet_mass


def convert_cluster_state_dict(state_dict):
    # Create a new state_dict without the "module." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    return new_state_dict


db = EventDataset(
    "../data/background.csv",
    "../data/signal.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    10_000,
    signal_proportion=0.01,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="multi_dim",
)
unconstrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
unconstrained_flow.load_state_dict(
    convert_cluster_state_dict(
        torch.load(
            "../saved_models_multi_dim_DEP/multi_dim_50_epochs_s0_b4096_p001.pth"
        )
    )
)
s_and_bg_densities = (
    unconstrained_flow.log_prob(db.features.detach()).exp().detach().numpy()
)

constrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
constrained_flow.load_state_dict(
    convert_cluster_state_dict(
        torch.load(
            "../saved_models_multi_dim_DEP/multi_dim_50_epochs_s05_b4096_p001.pth"
        )
    )
)
bg_densities = constrained_flow.log_prob(db.features.detach()).exp().detach().numpy()

likelihood_ratios = s_and_bg_densities / bg_densities
likelihood_ratios = likelihood_ratios
norm_likelihood_ratios = likelihood_ratios / likelihood_ratios.max()
signal_likelihood_ratios = likelihood_ratios[db.labels.flatten() == 1.0]
bg_likelihood_ratios = likelihood_ratios[db.labels.flatten() == 0.0]

# mass_axis = calculate_dijet_mass(db.features).detach().numpy()
# plt.scatter(mass_axis, norm_likelihood_ratios, label="Likelihood ratios")
# # plt.scatter(mass_axis, s_and_bg_densities, label="S+B density")
# # plt.scatter(mass_axis, bg_densities, label="B density")
# # plt.scatter(mass_axis, s_and_bg_densities - bg_densities, label="Density difference")
# plt.ylim(0.0, 0.4)
# plt.xlabel("Re-scaled mass")
# plt.ylabel("Re-scaled likelihood ratio")
# # plt.legend()
# plt.show()
# exit()

# Plot masses, labeled by prediction
threshold = 0.0001
limit = (0.0, 0.5)
bins = 50  # With more bins it looks a bit less promising...

real_signal_masses = calculate_dijet_mass(
    db.features.detach()[db.labels.flatten() == 1.0]
)
real_bg_masses = calculate_dijet_mass(db.features.detach()[db.labels.flatten() == 0.0])

pred_signal_masses = calculate_dijet_mass(
    db.features.detach()[norm_likelihood_ratios > threshold]
)
pred_bg_masses = calculate_dijet_mass(
    db.features.detach()[norm_likelihood_ratios <= threshold]
)

figure, axis = plt.subplots(1, 2, sharex=True, sharey=True)
axis[0].hist(
    [real_bg_masses, real_signal_masses],
    bins=bins,
    histtype="barstacked",
    color=["tab:blue", "tab:red"],
    range=limit,
)
axis[0].set_title("Actual")
axis[1].hist(
    [pred_bg_masses, pred_signal_masses],
    bins=bins,
    histtype="barstacked",
    color=["tab:blue", "tab:red"],
    range=limit,
)
axis[1].set_title("Predicted (multi-dim)")
figure.supxlabel("Normalized mass")
figure.supylabel("Entries")
figure.legend(["Background", "Signal"], loc="upper center")
plt.show()
