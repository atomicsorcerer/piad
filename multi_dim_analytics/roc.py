import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
import pandas as pd

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


unconstrained_model_path = (
    "../saved_models/fixed_loss_multi_dim_100_epochs_s00_b16384_p001_3.pth"
)
constrained_model_path = (
    "../saved_models/fixed_loss_multi_dim_100_epochs_s05_b16384_p001_9.pth"
)
save_file_name = "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_9.csv"

db = EventDataset(
    "../data/background.csv",
    "../data/signal.csv",
    ["energy_1", "px_1", "py_1", "pz_1", "energy_2", "px_2", "py_2", "pz_2"],
    50_000,
    signal_proportion=0.01,
    mass_region=(500.0, None),
    normalize=True,
    norm_type="multi_dim",
)
unconstrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
unconstrained_state_dict = torch.load(unconstrained_model_path)
unconstrained_state_dict = convert_cluster_state_dict(unconstrained_state_dict)
unconstrained_flow.load_state_dict(unconstrained_state_dict)
s_and_bg_densities = unconstrained_flow.log_prob(db.features.detach()).exp()

constrained_flow = create_spline_flow(10, 8, 32, 64, 4.0)
constrained_state_dict = torch.load(constrained_model_path)
constrained_state_dict = convert_cluster_state_dict(constrained_state_dict)
constrained_flow.load_state_dict(constrained_state_dict)
bg_densities = constrained_flow.log_prob(db.features.detach()).exp()

likelihood_ratios = s_and_bg_densities / bg_densities
likelihood_ratios = likelihood_ratios.detach().numpy()
norm_likelihood_ratios = likelihood_ratios / likelihood_ratios.max()
signal_likelihood_ratios = likelihood_ratios[db.labels.flatten() == 1.0]
bg_likelihood_ratios = likelihood_ratios[db.labels.flatten() == 0.0]

# Plot the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(db.labels.detach().numpy(), likelihood_ratios)
auc = metrics.roc_auc_score(db.labels.detach().numpy(), likelihood_ratios)

result = {
    "false_positive_rate": fpr,
    "true_positive_rate": tpr,
    "auc": [auc] * len(tpr),
}
result_df = pd.DataFrame(result)
result_df.to_csv("../results/" + save_file_name)

print(auc)

plt.plot(fpr, tpr, label="Our model (multi-dim)")
plt.plot(np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100), label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("Signal classification, batch size = 4096, s.p. = 1%, alpha = 1.0")
plt.legend()
plt.show()
