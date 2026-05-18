# EXP_009_CALIBRATION

## Hypothesis
- after surfacing uncertainty via MC dropout (EXP_008), we need to measure whether the model's predicted confidence actually matches its true accuracy per class
- ECE quantifies the calibration gap — if the model says 80% confident, is it actually correct 80% of the time?
- in a medical context, a well-calibrated model is more trustworthy than a high-accuracy but overconfident one
- per-class ECE should reveal which severity levels the model is most miscalibrated on



## Setup
- Loss: weightd Cross Entropy
- weights: [1805, 999, 370, 295, 193]
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 10
- proxies: train loss, val (validation) loss, confusion matrix, QWK score
- augmentation : resize (224, 224) -> flips and color jitter -> to tensor -> normalization
- MC dropout with droput = 0.3
- expected calibration error (ECE) per class

## Metrics
- train loss: 0.5
- val loss: 0.9
- QWK: 0.87

- Mean entropy      : 0.61
- Mean margin       : 0.588
- Mean MC std       : 0.03

- Class 0 (No DR) ECE: 0.0222
- Class 1 (Mild) ECE: 0.0551
- Class 2 (Moderate) ECE: 0.0768
- Class 3 (Severe) ECE: 0.0320
- Class 4 (Proliferative) ECE: 0.0317

## Observations
- Class 2 (Moderate) has the highest ECE at 0.0768 — aligns with every prior experiment showing Class 2 as the center of confusion
- Class 1 (Mild) is the second worst at 0.0551 — another class sitting in the ambiguous transition zone
- Class 0 (No DR) has the lowest ECE at 0.0222 — majority class with distinct features and abundant data
- Classes 3 and 4 have low ECE (0.032, 0.0317) despite being minority classes — severe/proliferative DR has more visually distinct features
- compared to EXP_008: entropy increased (0.457 → 0.61), margin decreased (0.70 → 0.588), MC std tripled (0.01 → 0.03) — model is expressing more uncertainty overall
- QWK held steady at 0.87, so ranking accuracy is preserved even as confidence distribution shifted

## Interpretation
- ECE confirms what the confusion matrices have shown since EXP_002: classes 1 and 2 sit in the transition zone of disease progression where visual differences are subtle and even experts disagree
- Class 0 is well calibrated because "no disease" has a distinct visual signature the model can learn with high confidence from abundant samples
- Classes 3 and 4 have lower ECE than 1 and 2 despite being minority classes — this means the visual features for severe DR (hemorrhages, neovascularization) are discriminative enough that the model can be genuinely confident, not just overconfident
- the increased entropy and MC std are actually desirable here — the model is expressing genuine uncertainty rather than hiding it behind a confident argmax, which is exactly what the clinical context requires
- per-class ECE gives actionable, class-specific information that aggregate metrics (QWK, loss) completely hide

## Conclusion
- calibration issues are class-specific: worst for middle severity classes (1, 2) which have the most ambiguous visual features and the most overlap in disease progression
- the model is reasonably well-calibrated overall (all ECE < 0.08) but Class 2 (Moderate) remains the weakest link — consistent with every experiment in this series