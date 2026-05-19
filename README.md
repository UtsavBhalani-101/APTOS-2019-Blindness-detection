# Diabetic Retinopathy Severity Classification

Automated grading of diabetic retinopathy from retinal fundus images into 5 severity levels (No DR → Proliferative DR), built on the APTOS 2019 dataset.

## Why This Is Hard

The dataset is simultaneously **ordinal**, **imbalanced**, and **noisy**. Disease severity is a continuum — adjacent classes share subtle visual differences that even ophthalmologists disagree on. Class distribution is heavily skewed (1805 No DR vs 193 Proliferative). Standard classification metrics hide dangerous failure modes: a model can achieve high accuracy by being confidently wrong on minority classes, which in a clinical screening context is the worst possible outcome.

## Approach

EfficientNet-B0 with weighted cross-entropy, trained iteratively across **9 experiments** — each driven by a specific diagnostic observation from the previous one, not by hyperparameter sweeping. The progression: baseline → validation-based evaluation → frequency-based class weights → augmentation for structural feature learning → MC Dropout for uncertainty quantification → per-class calibration measurement (ECE).

Key design decisions: focal loss was tested and rejected (hard samples in this dataset are noisy, not informative). Model capacity was validated as sufficient early — the bottleneck was generalization, not representation power.

## Results

- **QWK: 0.87** — ordinal ranking accuracy held stable from experiment 4 onward
- **Per-class ECE: 0.02–0.08** — calibration is worst on the middle severity classes (Mild, Moderate), confirming the transition-zone confusion pattern
- **Uncertainty quantification** via entropy, prediction margin, and MC Dropout std — the system flags which predictions are confident vs fragile

## Documentation

- [`experiments/Decision_log.md`](experiments/Decision_log.md) — why each technical choice was made
- [`experiments/Findings.md`](experiments/Findings.md) — cross-experiment analysis and recurring patterns
- [`experiments/EXP_001` through `EXP_009`](experiments/) — individual experiment logs with hypothesis, metrics, and interpretation
