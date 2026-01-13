from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from momfpriors.plotting.plot_styles import RC_PARAMS, other_fig_params

sns.set_theme(style="whitegrid")
sns.set_context("paper")


# Parameters
n_d = 4
beta = 10

n_bo = np.arange(0, 21, 1)  # Using the last value for n_bo

# Calculate γ = exp(-n²_bo/n_d)  # noqa: RUF003
# Note: This is constant with respect to n
gamma = np.exp(-(n_bo**2) / n_d)

# Calculate γ_n = β/n  # noqa: RUF003
gamma_n = beta / n_bo

# Create the plot
plt.figure(figsize=(12, 6))
plt.rcParams.update(RC_PARAMS)

# Plot both equations
plt.plot(n_bo, gamma, "b-", linewidth=2, label=r"$\gamma_{PriMO} = \exp(-n_{BO}^2/n_d)$")
plt.plot(n_bo, gamma_n, "r-", linewidth=2, label=r"$\gamma_{\pi{}BO} = \beta/n_{BO}$")

# Customize the plot
xylabel_fontsize = other_fig_params["xylabel_fontsize"]
legend_fontsize = other_fig_params["legend_fontsize"]

plt.xlabel(r"$n_{BO}$", fontsize=xylabel_fontsize)
plt.ylabel(r"$\gamma_{PriMO}$, $\gamma_{\pi{}BO}$ (log-scaled)", fontsize=xylabel_fontsize)
plt.title("Decay of $\gamma$ for PriMO and $\pi{}BO$ with growing number of BO samples")
plt.legend(fontsize=legend_fontsize)
plt.grid(visible=True, alpha=0.3)
plt.yscale("log")

plt.tight_layout()
save_path = Path("/home/soham/Master_Thesis/ICLR/plots/appendix/")
save_path.mkdir(parents=True, exist_ok=True)
plt.savefig(
    save_path / "gamma_decay_plot.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
