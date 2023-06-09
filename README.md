# Part 1


Below is the graph for different learning rates [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]:
![image](https://github.com/dknayakbu/ERA-V1-S6-Assignment/assets/20933037/44954277-ab24-425c-8e6f-7db79f57e12f)


# Part 2
# ERA-V1-Assignments for Session 6
This repository contains all the ERA V1 session 6 Assignments.

In this assignment for session 6, we are building a model to classify the MNIST dataset with the below constraints:
  - 99.4% validation accuracy
  - Less than 20k Parameters
  - You can use anything from above you want. 
  - Less than 20 Epochs
  - Have used BN, Dropout,
  - (Optional): a Fully connected layer, have used GAP. 

The Jupyter notebook having the solution is named "S6.ipynb".

# Results Achieved
We have achived the below results:
  - Total params: 11,952
  - Total Epochs: 20
  - Test set: Average loss: 0.0299, Accuracy: 9917/10000 (99.1700%)
  - We were able to achieve **99.1700% test accuracy** on 17th epoch.

# Model architecture details
We have used the below architecture to solve the problem:
The architecture can be divided in to 3 major sub-parts:

## 1. Model sub-part-1:
```
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 16, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25)
    )
```
Here we take convolution with output size 16 follwed by Relu activation follwed by Batch Normalisation. The same is repeated again but this time with convolution network having stride 2.
Next it is followed by a Maxpooling layer and a dropout layer which randomly masks 25% the neurons.

(Conv(16) -> ReLU -> BatchNorm(16)) -> (Conv(16, stride=2) -> ReLU -> BatchNorm(16)) -> MaxPool(2,2) -> Dropout(0.25)

## 2. Model sub-part-2:
```
    self.conv2 = nn.Sequential(
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 16, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25)
    )
```
The same network from the above is replicated again in sub-part 2.

(Conv(16) -> ReLU -> BatchNorm(16)) -> (Conv(16, stride=2) -> ReLU -> BatchNorm(16)) -> MaxPool(2,2) -> Dropout(0.25)

## 3. Model sub-part-3:
```
    self.conv3 = nn.Sequential(
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 16, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(0.25)
    )
```
Here the same network from above is repeated again, buyt this time without the Maxpooling layer at the end.

(Conv(16) -> ReLU -> BatchNorm(16)) -> (Conv(16, stride=2) -> ReLU -> BatchNorm(16)) -> Dropout(0.25)





