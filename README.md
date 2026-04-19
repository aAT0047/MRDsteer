# MRDsteer

**MRDsteer: quality-aware AI-driven closed-loop optimization enhances ctDNA-based minimal residual disease detection**

MRDsteer is an AI-driven closed-loop framework for adaptive ctDNA variant calling in minimal residual disease (MRD) analysis.  
It models latent performance degradation from multi-dimensional quality signals, predicts the remaining reliable processing span, and triggers localized recalibration before variant-calling failure occurs.

---

## Overview

Conventional ctDNA analysis pipelines are usually static and sequential. They run with predefined parameters and lack an explicit mechanism for:

- continuous monitoring of variant-calling degradation
- dynamic prediction of failure risk
- adaptive intervention in high-error regions

MRDsteer addresses this limitation by combining:

1. **Dynamic Error Modeling**  
   A non-linear Wiener process and Kalman filtering framework are used to estimate latent degradation and predict the remaining reliable processing span (RUL-like reliability horizon).

2. **Adaptive Recalibration**  
   A Double Deep Q-Network (DDQN) agent monitors degradation states and decides when to trigger localized rollback / re-calling for high-error genomic regions.

Together, these components transform ctDNA variant calling from a static pipeline into a **self-driving closed-loop optimization tool**.

---


## Key Features

- **Quality-aware degradation tracking** from accessible QC signals
- **Probabilistic state estimation** using Wiener-process-based modeling and Kalman filtering
- **RUL-style reliability prediction** for continuous monitoring
- **DDQN-based adaptive intervention** instead of fixed thresholds
- **Localized rollback / re-calling** to avoid costly global recalculation
- **Improved detection of ultra-low-frequency variants** in low-input ctDNA settings

---

## USING

MRDsteer consists of two major modules:

### 1. Dynamic Error Modeling Module

This module:

- reads sequential sample-level quality observations
- converts observed performance into a degradation quantity
- fits population-level degradation parameters
- estimates sample-specific latent states
- predicts the remaining reliable processing span

### 2. Adaptive Recalibration Module

This module:

- treats the estimated degradation state and predicted RUL as the environment state
- trains a DDQN agent to choose between:
  - **continue**
  - **localized rollback**
- learns an intervention policy that balances reliability risk and recalibration cost

---

## Repository Structure

```text
MRDsteer/
├── Dynamic Error Modeling Module.py
├── Adaptive Recalibration Module.py
├── manuscript.docx
├── README.md
└── data/
    ├── train.csv
    └── test.csv
```

> Suggested cleaner filenames:
>
> - `dynamic_error_modeling.py`
> - `adaptive_recalibration.py`

---

## Requirements

This project uses Python together with the following major packages:

- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `matplotlib`
- `torch`
- `jax`
- `jaxlib`
- `numpyro`

Example installation:

```bash
pip install numpy pandas scipy tqdm matplotlib torch numpyro
# install jax / jaxlib according to your platform (CPU or CUDA)
```

---

## Input Data Format

### A. Dynamic Error Modeling Module input

The dynamic modeling script expects CSV files containing at least:

- `sample_id`
- `row`
- `recall`

The script aggregates by `sample_id` and `row`, and defines:

- `X = 1 - recall`

### B. Adaptive Recalibration Module input

The DDQN training / evaluation script expects CSV files containing trajectories with at least:

- `sample_id`
- `X_pred`
- `rul`

where:

- `X_pred` represents the predicted degradation state
- `rul` represents the remaining reliable processing span

> If you want to run the full workflow end-to-end on your own data, make sure your preprocessing step converts outputs from the degradation module into the input format required by the DDQN module.

---

## Usage

### 1) Dynamic Error Modeling

This module fits degradation parameters from training data and predicts latent paths / RUL-related outputs for test data.

```bash
python "Dynamic Error Modeling Module.py" \
  --train_path data/train.csv \
  --test_path data/test.csv \
  --output_rul_path rul_predictions.csv \
  --output_lambda_path test_lambda_paths.csv \
  --failure_threshold 0.30 \
  --R_meas 0.000025
```

#### Main outputs

- `rul_predictions.csv`
- `test_lambda_paths.csv`

---

### 2) Adaptive Recalibration

This module trains a DDQN agent for intervention policy learning.

```bash
python "Adaptive Recalibration Module.py" \
  --train_file data/train.csv \
  --test_file data/test.csv \
  --episodes 50000 \
  --trainsteps 8 \
  --updatesteps 10000 \
  --batchsize 32 \
  --alpha 0.3
```

#### Main outputs

The script saves model and visualization files under the output directory, including:

- trained model weights
- training loss curve
- cumulative reward curve
- epsilon decay curve
- suggested rollback point visualizations for train/test samples

---

## Method Summary

### Dynamic Error Modeling

MRDsteer formulates variant-calling deterioration as a latent degradation process.  
A non-linear Wiener process models performance drift, while Kalman filtering reconstructs hidden degradation states from noisy quality observations.

This module provides:

- cohort-level parameter estimation
- sample-specific latent-state inference
- reliability boundary prediction
- first-passage-time-based RUL estimation

### Reinforcement Learning Control

The adaptive controller uses a DDQN agent with experience replay.  
At each step, the state is represented by:

- degradation estimate
- predicted remaining reliable processing span

The action space is binary:

- **0** = continue
- **1** = rollback

The reward is defined as the negative system cost, encouraging the policy to intervene before catastrophic failure while avoiding excessive recalibration.

---

## Results Highlight

According to the manuscript, MRDsteer achieved strong performance under analytically challenging ctDNA conditions:

- **AUPRG = 0.76**
- **F1-minor = 0.86**

under ultra-low variant allele frequency conditions (**0.1%–0.5% VAF**) with **10 ng DNA input**.

The manuscript also reports improved progression-free survival stratification in clinical cohorts:

- **K438:** HR = 2.63, log-rank *P* = 0.003
- **KA7P:** HR = 2.41, log-rank *P* = 0.012

---

## Notes

- This repository currently exposes the two main computational modules corresponding to the paper.
- Depending on your local dataset format, you may need an additional preprocessing step to connect the degradation outputs with the reinforcement learning inputs.
- The current scripts are most suitable for reproducing the core methodology and adapting it to structured ctDNA quality trajectories.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{mrdsteer2026,
  title={MRDsteer: quality-aware AI-driven closed-loop optimization enhances ctDNA-based minimal residual disease detection},
  author={...},
  journal={...},
  year={2026}
}
```

> Please replace the author, journal, and year fields with the final publication information.

---

## License

Please add your preferred open-source license here, for example:

- MIT License
- Apache-2.0
- GPL-3.0

---

## Contact

For questions, collaborations, or issues, please open a GitHub Issue or contact the authors.
