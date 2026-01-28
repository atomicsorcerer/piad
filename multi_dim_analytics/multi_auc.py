import pandas as pd
import matplotlib.pyplot as plt


csv_to_import = [
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_0.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_1.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_2.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_3.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_4.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_5.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_6.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_7.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_8.csv",
    "fixed_loss_multi_dim_100_epochs_s05_b16384_p001_9.csv",
]
labels = [
    "Ensemble 1",
    "Ensemble 2",
    "Ensemble 3",
    "Ensemble 4",
    "Ensemble 5",
    "Ensemble 6",
    "Ensemble 7",
    "Ensemble 8",
    "Ensemble 9",
    "Ensemble 10",
    # "alpha = 0.1",
    # "alpha = 0.25",
    # "alpha = 0.5",
    # "alpha = 0.75",
    # "alpha = 1.0",
    # "alpha = 10.0",
]

fprs = []
tprs = []

for file, label in zip(csv_to_import, labels):
    csv = pd.read_csv("../results/" + file)
    plt.plot(
        csv["false_positive_rate"].to_list(),
        csv["true_positive_rate"].to_list(),
        label=label,
    )

plt.plot([0.0, 1.0], [0.0, 1.0], label="random", linestyle="dashed", c="black")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("Batch size = 4096, Signal proportion = 1%, alpha=0.1")
plt.legend()
plt.show()
