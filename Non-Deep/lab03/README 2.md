# Introduction to Natural Language Processing 01
# Lab 03

## Introduction

The project is a continuation of what we started on the second lab. 

## Features

### Loading datasets
We retrieve Imdb datasets and create train and test datasets.
### Features extraction 
- For every given text we generate a vector with specific features in order to generate data frames for train and test.
- Our bonus feature is the third person pronoun count. 
## Logistic regression classifier
- We use the logistic resgression class from `logistic_regression_pytorch.ipynb` notebook. 
- We generate a model with 8 input dimensions (7 features + Bonus Feature) and 1 class (negative or positive)
  - Its loss function is based on binary cross entropy.
  - Its optimizer is stochastic gradient descent.

- We fit our model.
- We display our training and validation losses. 
# Project tree

lab03
 * [lab03.ipynb](lab03/lab03.ipynb)
 * [README.md](lab03/README.md)
