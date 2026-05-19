

1. why choose cross entropy and why switch to WCE ?
- the task is a classification task and the first standard loss fuction is CE
- the data we are dealing with is ordinal + imbalance
- the CE assumes every class is equally important, to fix that weighted CE is used

2. Why added augmentation and specifically what is the reasoning behind each of those augmentations ?
- the task is to force the model to learn underlying structural patterns and not make decisions based on pixel level information 
- the another reason to add augmentation is this model is also going to be tested on different datasets, to capture cross dataset features accurately and to make it robust 

- the normal flow of augmentation is image transformations -> convert to tensor -> normalization
- each transformation is made with a reason that it should force model to not make predictions from pixels _and_ also not lose information as medical data is sensitive 

3. Why not use focal loss ?
- ans in  [Findings](Findings.md)

4. Why used efficient net model b0 ?
- it's fast and efficient, the b0 version (smallest in the family) is used in the start to quickly test the scope and difficulty of the task 
- it performed better then expected and there was no need to change the model, as the model was not a problem but representation and data
- when testing on other dataset, I will change it to a bigger model when I will be testing other datasets and if it's required
- Training loss reached 0.5 within 5-10 epochs and overfit to near 0.0, indicating the model has sufficient capacity to learn the data. The bottleneck was generalization, not model capacity — confirmed by val loss plateauing while train loss continued dropping. Switching to a larger model would worsen overfitting, not improve generalization

5. What's the goal and why ?
- The goal is not just to get high numbers, the goal is to make this system robust enough that if tomorrow this is deployed in a clinic, it should handle the messy real world and be grounded by the things it actually learned and not overconfident
- this is also a part of the reason why I am added different metrics, testing on other datasets, showing where the model is uncertain and optimizing 


6. Why used MC dropout ?
- ans in [Findings](Findings.md)