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

analysis = (
    data[variables]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
analysis = analysis.fillna(analysis.mean())
x_scaled = StandardScaler().fit_transform(analysis)

# Full PCA
pca_full = PCA().fit(x_scaled)
eigenvalues = pca_full.explained_variance_
explained = pca_full.explained_variance_ratio_
cumulative = explained.cumsum()
n_kaiser = (eigenvalues > 1).sum()

# Two-component PCA
pca = PCA(n_components=2)
scores = pca.fit_transform(x_scaled)
loadings = pca.components_.T


# Varimax rotation
def varimax(x, gamma=1.0, max_iter=100, tol=1e-6):
    n, k = x.shape
    rotation, previous = np.eye(k), 0

    for _ in range(max_iter):
        rotated = x @ rotation
        u, values, vh = np.linalg.svd(
            x.T @ (
                rotated**3
                - gamma / n * rotated
                @ np.diag(np.diag(rotated.T @ rotated))
            )
        )
        rotation = u @ vh
        current = values.sum()

        if previous and current / previous < 1 + tol:
            break
        previous = current

    return x @ rotation, rotation


loadings, rotation = varimax(loadings)
scores = scores @ rotation

scores_df = pd.DataFrame(scores, index=analysis.index, columns=["RC1", "RC2"])
loadings_df = pd.DataFrame(loadings, index=variables, columns=["RC1", "RC2"])

if "Messstelle" in data:
    scores_df.insert(0, "Messstelle", data.loc[analysis.index, "Messstelle"])

scores_df.round(4).to_csv("pca_varimax_scores.csv", index=False)
loadings_df.round(4).to_csv("pca_varimax_loadings.csv")

pd.DataFrame({
    "component": [f"PC{i}" for i in range(1, len(variables) + 1)],
    "eigenvalue": eigenvalues,
    "explained_variance_ratio": explained,
    "cumulative_variance_ratio": cumulative,
}).round(4).to_csv("pca_explained_variance.csv", index=False)


# Scree plot
components = np.arange(1, len(variables) + 1)
fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(components, explained * 100, width=0.45)
ax.plot(components, explained * 100, marker="o")

ax2 = ax.twinx()
ax2.plot(components, cumulative * 100, marker="o")

ax.text(
    0.68, 0.70, f"Kaiser criterion: {n_kaiser} PCs",
    transform=ax.transAxes,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
)

ax.set(
    xlabel="Principal component",
    ylabel="Explained variance (%)",
    xticks=components,
    ylim=(0, 100),
)
ax2.set(ylabel="Cumulative explained variance (%)", ylim=(0, 100))

fig.tight_layout()
fig.savefig("pca_scree_plot.pdf", dpi=300, bbox_inches="tight")
fig.savefig("pca_scree_plot.svg", bbox_inches="tight")
plt.close(fig)


# Rotated loading heatmap
fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    loadings_df, annot=True, fmt=".2f", cmap="Spectral",
    center=0, vmin=-1, vmax=1, ax=ax,
)

ax.set(
    title="Varimax-rotated PCA loadings",
    xlabel="Rotated component",
    ylabel="",
)

fig.tight_layout()
fig.savefig("pca_varimax_loadings.pdf", dpi=300, bbox_inches="tight")
fig.savefig("pca_varimax_loadings.svg", bbox_inches="tight")
plt.close(fig)

print(f"Analysed observations: {len(analysis)}")
print(f"Kaiser criterion: {n_kaiser} components")
print(f"Explained variance PC1-PC2: {explained[:2].sum() * 100:.1f}%")

