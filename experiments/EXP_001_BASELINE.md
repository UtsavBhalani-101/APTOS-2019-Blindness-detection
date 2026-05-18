# EXP_001_BASELINE_CE

## Hypothesis
- This would either overfit and numbers would be inacurate


## Setup
- Loss: Cross Entropy
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 5
- just normal pipeline -- train dataloader, running training loop

## Metrics
- Train Loss: 0.07

## Observations
- Loss suspiciously low.
- Dataset highly ordinal + imbalanced.

## Interpretation
- Model rapidly fitting dominant signal.
- Possible early memorization risk.

## Conclusion
Need validation-based evaluation.
Training loss alone insufficient.