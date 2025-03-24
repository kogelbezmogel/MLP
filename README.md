# Multilayer Perceptron library

This library enables users to create and train multilayer preceptron models in C++. Usage is similar to pytorch library for python.
The example usage can be seen in main.cpp where MLP model is trained on Iris Dataset. The main classes in library are:
+ Dataset - to load dataset from csv file
+ Dataloader - to load data in batches from dataset
+ Model - to represent MLP model
+ Optimizer - to represent learning algorithm BGD or SGD
  
Calculation are made with my implementiation of tensors and automatic differentiation mechanism.


# Learning results for Iris Dataset
+ Dataset \
  Dataset is a classification problem with 3 classes. There are 150 samples with 4 attributes and label each.
   
+ Model \
  Created model is represented in the diagram below
  <p>
  <img src="https://github.com/kogelbezmogel/MLP/blob/master/model_architecture.png" width="160" height="300"/>
  </p>

+ Learning results
<p float="left">
  <img src="https://github.com/kogelbezmogel/MLP/blob/master/training_loss.png" width="350" height="350"/>
  <img src="https://github.com/kogelbezmogel/MLP/blob/master/training_accuracy.png" width="350" height="350"/>
</p>
  

# Learning and loss visualisation for only 2 parameters.
+ Regression. Problem was random points from line y = ax + b with some noise. Stochastic Gradient Descent (left) vs Batch Gradient Descent (right)
<p float="left">
  <img src="https://github.com/kogelbezmogel/MLP/blob/master/learning_regression_sgd.gif" width="350" height="350"/>
  <img src="https://github.com/kogelbezmogel/MLP/blob/master/learning_regression_bgd.gif" width="350" height="350"/>
</p>

+ Classification. Problem was distinction between two distributions N([], []) and N([], []). Stochastic Gradient Descent (left) vs Batch Gradient Descent (right)
<p float="left">
  <img src="https://github.com/kogelbezmogel/MLP/blob/master/learning_classification_sgd.gif" width="350" height="350"/>
  <img src="https://github.com/kogelbezmogel/MLP/blob/master/learning_classification_bgd.gif" width="350" height="350"/>
</p>
