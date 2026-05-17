# EXP_002_VALIDATION

## Goal
Test the model on validation dataset and QWK (quadratic weighted kappa) score

## Hypothesis
- These might be better proxies compared to training loss and reval actual problem


## Setup
- Loss: Cross Entropy
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 5
- proxies: train loss, val (validation) loss, confusion matrix, QWK score
 

## Metrics
- train loss = 0.07
- val loss = 0.81
- QWK score = 0.83

## Observations
- The gap beteween loss is pretty big 
- QWK score is suspiciously high
- most predictions gravitate towards class 2 and 0 

## Interpretation
- Model is overfitting = generalizable signals + noise memorization
- it's performing great on majority classes but bad on minority and adaject classes and most of the error are related to adjacent class - target is class 3 and model is saying class 2
- The data is imbalanced so the model is treating all the classes with equal weights (might be a problem)

## Conclusion
Try giving weights to target 

>  Note: the number of epochs are less because :
> 1. the loss was converging faster then expected
> 2. it was taking too long to run and get feedback