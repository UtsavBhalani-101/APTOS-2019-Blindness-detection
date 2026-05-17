# EXP_003_WEIGHTS

## Hypothesis
- just give weights in the order of severity of that class error problem [1,2,3,4,5] 
- it's ordered 

## Goal
Test weighted 

## Setup
- Loss: weightd Cross Entropy
- weights: [1,2,3,4,5]
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 5
- proxies: train loss, val (validation) loss, confusion matrix, QWK score

## Metrics
- train loss: 0.05
- val loss: 1.2
- QWK: 0.82

## Observations
- even after implementing the weighted loss, the numbers decreased
- even after bad weights, QWK barely changed

## Interpretation
- the model is learning signals and somwhat robust but bad weights hurts performace
- weighted CE expects frequency based weights and not order - based
- the weights above are not capturing the real imbalance at all

## Conclusion
fix the weights 