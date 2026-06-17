# Frequency-Aware NAS with Bidirectional Mamba for EEG Emotion Recognition

This repository contains the implementation of a frequency-aware neural architecture search framework with Bidirectional Mamba modules for DEAP EEG emotion recognition.

The code release is organized for a paper-style GitHub repository. DEAP data are not included and must be obtained from the official dataset provider. The scripts assume preprocessed DEAP `.mat` files named `s01.mat` to `s32.mat` with the standard DEAP fields `data` and `labels`.

## Repository structure

```text
frequency_aware_bimamba_nas/
├── src/
│   ├── bidirectional_mamba.py       # Bidirectional Mamba blocks
│   ├── controller.py                # RL controller for architecture sampling
│   ├── dag_search_space.py          # DAG search space and frequency fusion modules
│   ├── deap_dataset.py              # DEAP DE preprocessing and datasets
│   ├── nas_search.py                # Single-band architecture search
│   ├── nas_eval.py                  # LOSO evaluation using searched architectures
│   ├── spatial_feature_extractor.py # EEG spatial graph feature extractor
│   └── utils.py                     # Loss and architecture I/O utilities
├── scripts/
│   ├── run_preprocess.sh
│   ├── run_search.sh
│   └── run_eval.sh
├── configs/default.yaml
├── requirements.txt
└── .gitignore
```

## Installation

```bash
conda create -n bimamba_nas python=3.10 -y
conda activate bimamba_nas
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` are CUDA/PyTorch-version sensitive. Install the versions compatible with your own CUDA and PyTorch environment.

## Data preparation

Expected raw DEAP layout:

```text
/path/to/DEAP_Preprocessed_EEG/
├── s01.mat
├── s02.mat
└── ...
```

Generate band-wise differential entropy features:

```bash
bash scripts/run_preprocess.sh /path/to/DEAP_Preprocessed_EEG ./data/DEAP valence
```

For arousal:

```bash
bash scripts/run_preprocess.sh /path/to/DEAP_Preprocessed_EEG ./data/DEAP_arousal arousal
```

The output directory will contain four band folders: `theta`, `alpha`, `beta`, and `gamma`.

## Architecture search

```bash
bash scripts/run_search.sh ./data/DEAP ./results
```

This searches one architecture per frequency band and writes:

```text
results/best_arch_theta.pth
results/best_arch_alpha.pth
results/best_arch_beta.pth
results/best_arch_gamma.pth
```
## LOSO evaluation

```bash
bash scripts/run_eval.sh ./data/DEAP ./results
```

The final evaluation script uses the searched band architectures and performs leave-one-subject-out evaluation. Within each LOSO fold, the held-out test subject is excluded from both training and validation.

Outputs include:

```text
results/final_results_crossattn_eval.json
results/final_results_crossattn_eval.pth
```

A temporary `loso_checkpoint.pth` is written during evaluation for resuming interrupted runs and removed after completion.

## Important protocol note

The provided implementation separates architecture search and final LOSO evaluation as follows:

1. Search: fixed subject-level training/validation split.
2. Evaluation: LOSO final evaluation with per-fold validation subjects drawn only from the training subjects.

## Reproducibility notes

Random seeds are set in the search and evaluation scripts. Exact numerical reproduction can still vary across GPUs, CUDA versions, PyTorch versions, and `mamba-ssm` kernels.

## Minimal command sequence

```bash
conda activate bimamba_nas
bash scripts/run_preprocess.sh /path/to/DEAP_Preprocessed_EEG ./data/DEAP valence
bash scripts/run_search.sh ./data/DEAP ./results
bash scripts/run_eval.sh ./data/DEAP ./results
```
