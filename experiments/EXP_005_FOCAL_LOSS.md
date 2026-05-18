# EXP_005_FOCAL_LOSS

## Hypothesis
-  the data is ordinal + imbalance, try focal loss which forces model to learn hard and confusing samples instead of easy ones
- this should atleast now decrease the results 


## Setup
- Loss: focal loss (gamma: 1.0)
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 5
- proxies: train loss, val (validation) loss, confusion matrix, QWK score

## Metrics
- train loss and val loss: similar to previos
- QWK: 0.81

## Observations
- forcing the model to learn the hard samples is performing worse

## Interpretation
- focal loss is made optimizing for hard samples which might be not that informative - like blurry, noisy, low quality, etc
- the resutls here are not clearly seperable, the data contains label noise and even experts might diagree on severity levels between adjacent classes

## Conclusion
- focal loss is not suited for this dataset — the hard samples it prioritizes are ambiguous/noisy rather than informative
- the adjacent class confusion is still the core issue, and it's a data problem (not enough discriminative samples) not a loss function problem
- try augmentation next to give the model more varied views of minority classes before changing loss strategy further


> NOTE: before this I tried focal loss with weights defined which is essentially double impact and that was not recorded but the result didn't had any significant changes and it's also not correct to double count 

