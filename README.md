# project Tranfer Learning

[TOC]

## environment

python3.7.4 with VScode on windows10, colab

## run

Open source code in a python IDE and run.

Or open the following colab link to view and run the code.

<https://colab.research.google.com/drive/1CwXJ9SNm4rZQhvDl4vhlCSk7EaJkTlMP>

## Step 1

Create our own Neural Net for the Hepatitis Dataset with specific set of parameters for layers

## Step 2

In the Neural Net, we assign random weights to matrices,

The number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)

## Step 3

Split the Dataset with 80% for training and 20% for testing, with max_iterations=100000 and learning_rate=0.001

## Step 4

Get the accuracy for the Neural Net from the Hepatitis Dataset with random weights

## Step 5

Use keras framework to find the weights for the same set of parameters for layers of Breast Cancer Dataset

## Step 6
a. First, select the same number of Hepatitis' features as in the Breast Cancer Dataset, and split the dataset as 80% for training and 20% for testing

b. Second, transfer all the layers' weights include w12, w23, w34 for the training and get the accuracy

c. Third, transfer each layer's weights (such as w12, w23, w34) for the traiing and get the accuracy


## dataset

We use UCI Machine Learning Repository. “Breast Cancer Dataset”. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer, 9 features, 221 instance

And UCI Machine Learning Repository. “Hepatitis Dataset”. https://archive.ics.uci.edu/ml/datasets/hepatitis, 14 features, 155 instance

## result

We go through step 1 to step 6.b 2 times for different set of parameters for the layers

First set is 60 (relu),30 (relu),15 (relu),1 (tanh),1 (sigmoid), the result of step 6.b is 0.322

Second set is 10 (sigmoid),5 (sigmoid),5(sigmoid),1 (sigmoid), the result of step 6.b is 0.741935

As a reslut, we choose set with 10 (sigmoid),5 (sigmoid),5(sigmoid),1 (sigmoid) for step 6.c

In step 6.c the accuracy of transferring only w12 is 0.677419, only w23 is 0.741935, and only w34 is 0.709677



