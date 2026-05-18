# EXP_007_MC_DROPOUT

## Hypothesis
- in this context of this dataset where we want to know confidence of each model's predictions because confident wrong preds are dangerous
- MC dropout help in surfacing those issues 
- also adding new better proxies (entropy and mean) to get model's prediction uncertainty 


## Setup
- Loss: weightd Cross Entropy
- weights: [1805, 999, 370, 295, 193]
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 10
- proxies: train loss, val (validation) loss, confusion matrix, QWK score
- augmentation : resize (224, 224) -> flips and color jitter -> to tensor -> normalization
- MC dropout with droput = 0.3

## Metrics
- train loss: 0.5
- val loss: 0.8
- QWK: 0.866
- Mean entropy      : 0.4570
- Mean margin       : 0.7005
- Mean MC std       : 0.0102


## Observations
- the model's prediction uncertainty is observed by entropy and mean for one forward pass and it went from 80% to 22% 
- the other metrics (val loss 0.8, QWK 0.866) improved compared to EXP_007 — MC dropout also acts as regularization during training, reducing co-adaptation and improving generalization

## Interpretation
- the numbers improved partly due to MC dropout also applies regularization during training which as a side effect improves the numbers
- surfacing fragile confidence which flips under perturbation 

## Conclusion
- MC dropout served two purposes: regularization (improved numbers) and uncertainty surfacing (entropy, margin, MC std)
- the model's predictions can now be flagged as uncertain vs confident, which is critical for the clinical deployment context
- next step: measure whether the model's confidence is actually calibrated per class using ECE — knowing uncertainty exists is useful, knowing if the confidence scores are trustworthy is actionable