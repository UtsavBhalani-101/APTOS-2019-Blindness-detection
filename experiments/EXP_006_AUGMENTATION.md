# EXP_006_AUGMENTATION

## Hypothesis
-  the model was making errors on adjacent classes -> don't have enough data and discriminative patterns to distinguish so we use augmentation to fix it


## Setup
- Loss: focal loss
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 5
- proxies: train loss, val (validation) loss, confusion matrix, QWK score
- augmentation : resize (224, 224) -> flips and color jitter -> to tensor -> normalization

## Metrics
- train loss: ~0.7
- val loss: ~1.2
- QWK: 0.76

## Observations
- image augmentation + focal loss lead to significantly worse performance
- the QWK score has seen the lowest drop 

## Interpretation
- the focal loss forces the model on hard samples, it's not necessary the hard samples are informative here, because of augmentation, these hard samples might be just noisy, blurry images which model would get wrong anyways 

## Conclusion
switch back to weighted CE because focal loss is not helping at all