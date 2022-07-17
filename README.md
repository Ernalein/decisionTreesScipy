# decisionTreesScipy

- using libraries numpy, pandas, matplotlib
- function that takes dataset (pandas data frame) and returns a trained decision tree
- decision tree is realised recursively (class node)
- a method that prints the decision tree
- a method that returns classification of data point
- optional parameters to avoid overfitting (C4.5)?
- a method that prunes the tree ?
- a method that plots some accuracy overfitting whatever
- decison forest ?

### How to train a tree

- input: whole training set, classification column
- calculate entropy and information gain for each attricbute
- choose atribute with highest information gain
- divide dataset into subsets
- recursively call nodeclass for each subset
