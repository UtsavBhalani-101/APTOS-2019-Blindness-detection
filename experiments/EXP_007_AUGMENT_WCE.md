# EXP_007_AUGMENTATION_WCE

## Hypothesis
-  the score should recover back and should slightly improve because augmentation is added 


## Setup
- Loss: weightd Cross Entropy
- weights: [1805, 999, 370, 295, 193]
- Dataset: APTOS
- model: efficientnet_b0
- epochs: 10
- proxies: train loss, val (validation) loss, confusion matrix, QWK score
- augmentation : resize (224, 224) -> flips and color jitter -> to tensor -> normalization

## Metrics
- train loss: 0.5
- val loss: 0.9
- QWK: 0.859

## Observations
- as hypothsized, the numbers improved
- the confusion matrix for class 2 error is spread all over, same for class 3, 4 and 1
- these classes are either predicting the correct class as 1st priority or class 2 as 2nd priority

## Interpretation
- the model is now forced to learn structural feature invariants and more discriminative features that differentiate between classes

## Conclusion
- the WCE + augmentation is the right move 
- now the issues is that these overdependence on class 2 and bi directional errors


----------------------------------
Training loss calculated for epoch: 9 = 0.5772
----------------------------------
Val Loss: 0.9626 | Val QWK: 0.8597
Confusion Matrix: 
 [[343  14   3   0   1]
 [  7  46  21   0   0]
 [  1  30 122  24  23]
 [  0   1   8  20  10]
 [  0   9   7   8  35]]
Epoch 10 | Train Loss: 0.5772 | Val QWK: 0.8597
