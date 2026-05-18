# EXP_004_FIXED_WEIGHTS

## Hypothesis
- replacing the weights with their value counts should improve the numbers 
- and help give equal priority to all classes



## Setup
- Loss: weightd Cross Entropy
- weights: [1805, 999, 370, 295, 193]
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 5
- proxies: train loss, val (validation) loss, confusion matrix, QWK score

## Metrics
- train loss: 0.1
- val loss: 1.6
- QWK: 0.82

## Observations
- the numbers got worse even but based on the confusion matrix, it's giving more priority to the minority classes 

## Interpretation
- because the model is forced to predict minority classes on which it don't have enough data, it's making more errors 
- the model is probably learning a narrow set of features for the minority classes that are not robust and when tested, it always predicts that class 

## Conclusion
model is robust and overfitting
- QWK mainly measures if order is maintained and CE punishes confidence calibration
- the order is preserved as QWK is high and barely moving, the calibration is low 