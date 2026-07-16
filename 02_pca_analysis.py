#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Principal component analysis of selected MultiPAC variables
and EU-WFD indicators.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


variables = ["IDO_oIP", "kf_oIP", "FS < 2.0", "Fr_I", "S_I", "ES", "GD"]

analysis = data[variables].apply(pd.to_numeric, errors="coerce")
analysis = analysis.replace([np.inf, -np.inf], np.nan)
analysis = analysis.fillna(analysis.mean())

x_scaled = StandardScaler().fit_transform(analysis)

# Full PCA for explained variance and Kaiser criterion
pca_full = PCA().fit(x_scaled)
eigenvalues = pca_full.explained_variance_
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_kaiser = int(np.sum(eigenvalues > 1))

# Two-component PCA
pca = PCA(n_components=2)
scores = pca.fit_transform(x_scaled)
loadings = pca.components_.T

scores_df = pd.DataFrame(scores, columns=["PC1", "PC2"], index=data.index)
loadings_df = pd.DataFrame(loadings, index=variables, columns=["PC1", "PC2"])

if "Messstelle" in data.columns:
    scores_df.insert(0, "Messstelle", data["Messstelle"])

scores_df.to_csv("pca_scores.csv", index=False)
loadings_df.round(4).to_csv("pca_loadings.csv")

explained_df = pd.DataFrame({
    "component": [f"PC{i}" for i in range(1, len(variables) + 1)],
    "eigenvalue": eigenvalues,
    "explained_variance_ratio": explained,
    "cumulative_variance_ratio": cumulative,
})
explained_df.round(4).to_csv("pca_explained_variance.csv", index=False)

# Scree plot
components = np.arange(1, len(variables) + 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(components, explained * 100, width=0.45)
ax.plot(components, explained * 100, marker="o")

ax2 = ax.twinx()
ax2.plot(components, cumulative * 100, marker="o")

ax.text(0.68, 0.70, f"Kaiser criterion: {n_kaiser} PCs",
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})

ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance (%)")
ax2.set_ylabel("Cumulative explained variance (%)")
ax.set_xticks(components)
ax.set_ylim(0, 100)
ax2.set_ylim(0, 100)

fig.tight_layout()
fig.savefig("pca_scree_plot.pdf", dpi=300, bbox_inches="tight")
fig.savefig("pca_scree_plot.svg", bbox_inches="tight")
plt.close(fig)

# Loading heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(loadings_df, annot=True, fmt=".2f", cmap="Spectral",
            center=0, vmin=-1, vmax=1, ax=ax)

ax.set_title("PCA component loadings")
ax.set_xlabel("Principal component")
ax.set_ylabel("")

fig.tight_layout()
fig.savefig("pca_loadings.pdf", dpi=300, bbox_inches="tight")
fig.savefig("pca_loadings.svg", bbox_inches="tight")
plt.close(fig)

print(f"Analysed observations: {len(analysis)}")
print(f"Kaiser criterion: {n_kaiser} components")
print(f"Explained variance PC1–PC2: {explained[:2].sum() * 100:.1f}%")

