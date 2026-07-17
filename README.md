# Riverbed-Clogging-Classification

Reproducible Python workflows accompanying the MultiPAC framework for riverbed clogging assessment.

**Aybar-Galdos et al.**
*Towards the identification of reference values for riverbed clogging using selected abiotic parameters and standard indicators of the European Water Framework Directive*

## Repository structure

```
Riverbed-Clogging-Classification
│
├── scripts/
│   ├── 01_spearman_analysis.py
│   └── 02_pca_analysis.py
│
├── data/
│
├── results/
│
├── figures/
│
└── requirements.txt
```

## Statistical analyses

The repository contains the Python scripts used for:

- Spearman correlation analysis
- Principal Component Analysis (PCA)
- Varimax rotation
- Generation of publication-quality figures

## Input data

The scripts require a processed dataset containing the variables described in the manuscript.

The original field data belong to an ongoing research project and are available from the corresponding author upon reasonable request.

## Requirements

```
numpy
pandas
matplotlib
scipy
scikit-learn
seaborn
```

## Citation

If you use this repository, please cite the associated publication.
