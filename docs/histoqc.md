# Optional HistoQC environment

This env installs HistoQC from our fork (compat fixes for scikit-image 0.22) and pins a known-good stack on Apple Silicon.

## Create and activate
```bash
mamba env create -f envs/pyslyde-histoqc.yml
mamba activate pyslyde-histoqc
