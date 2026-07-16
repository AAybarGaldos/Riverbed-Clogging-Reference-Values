#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Spearman correlation analysis between MultiPAC variables
and EU-WFD indicators.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


variables = [
    "IDO_M", "IDO_oIP", "IDO_uIP", "IDO_15-20",
    "kf_M", "kf_oIP", "kf_uIP", "kf_15-20",
    "d16", "d84", "dm",
    "FS < 0.5", "FS < 1.0", "FS < 2.0",
    "Fr_I", "S_I", "sigma_geo", "n_cal",
    "ES", "BMI", "GD", "MH", "BS",
]

analysis = data[variables].apply(pd.to_numeric, errors="coerce")
analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna()

rho, p_values = spearmanr(analysis, axis=0)

rho_df = pd.DataFrame(rho, index=variables, columns=variables)
p_df = pd.DataFrame(p_values, index=variables, columns=variables)

filtered = rho_df.where((rho_df.abs() > 0.30) & (p_df < 0.05))
np.fill_diagonal(filtered.values, 1.0)

rho_df.round(3).to_csv("spearman_correlations.csv")
p_df.round(4).to_csv("spearman_p_values.csv")

mask = np.triu(np.ones(filtered.shape, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(filtered, mask=mask, cmap="Spectral", center=0, vmin=-1, vmax=1,
            square=True, annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"label": "Spearman correlation coefficient (rho)"}, ax=ax)

ax.set_title("Significant Spearman correlations (|rho| > 0.30, p < 0.05)")
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=90)
ax.tick_params(axis="y", rotation=0)

fig.tight_layout()
fig.savefig("spearman_heatmap.pdf", dpi=300, bbox_inches="tight")
fig.savefig("spearman_heatmap.svg", bbox_inches="tight")
plt.close(fig)

print(f"Analysed observations: {len(analysis)}")

