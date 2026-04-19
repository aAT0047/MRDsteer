# MRDsteer

**MRDsteer: quality-aware AI-driven closed-loop optimization enhances ctDNA-based minimal residual disease detection**

MRDsteer is an AI-driven closed-loop framework for adaptive ctDNA variant calling in minimal residual disease (MRD) analysis.  
It models latent performance degradation from multi-dimensional quality signals, predicts the remaining reliable processing span, and triggers localized recalibration before variant-calling failure occurs.

---

## Overview

Conventional ctDNA analysis pipelines are usually static and sequential. They run with predefined parameters and lack an explicit mechanism for:

- continuous monitoring of variant-calling degradation,
- dynamic prediction of failure risk,
- adaptive intervention in high-error regions.

MRDsteer addresses this limitation by combining:

1. **Dynamic Error Modeling**  
   A non-linear Wiener process and Kalman filtering framework are used to estimate latent degradation and predict the remaining reliable processing span (RUL-like reliability horizon).

2. **Adaptive Recalibration**  
   A Double Deep Q-Network (DDQN) agent monitors degradation states and decides when to trigger localized rollback / re-calling for high-error genomic regions.

Together, these components transform ctDNA variant calling from a static pipeline into a **self-driving closed-loop optimization framework**.

---

## Key Features

- **Quality-aware degradation tracking** from accessible QC signals
- **Probabilistic state estimation** using Wiener-process-based modeling and Kalman filtering
- **RUL-style reliability prediction** for continuous monitoring
- **DDQN-based adaptive intervention** instead of fixed thresholds
- **Localized rollback / re-calling** to avoid costly global recalculation
- **Improved detection of ultra-low-frequency variants** in low-input ctDNA settings

---

## Framework

MRDsteer consists of two major modules:

### 1. Dynamic Error Modeling Module

This module:

- reads sequential sample-level quality observations,
- converts observed performance into a degradation quantity,
- fits population-level degradation parameters,
- estimates sample-specific latent states,
- predicts the remaining reliable processing span.

### 2. Adaptive Recalibration Module

This module:

- treats the estimated degradation state and predicted RUL as the environment state,
- trains a DDQN agent to choose between:
  - **continue**, or
  - **localized rollback**
- learns an intervention policy that balances reliability risk and recalibration cost.

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
