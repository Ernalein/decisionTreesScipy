## Decision Trees project

We implemented the classical id3 algorithm to train decsion trees in the Train_data class.
The Node class implements the resulting tree structure in a recursive way.
We also implemented some extra features in our training algorithm:
1.  Instead of using just the Information Gain for choosing the attributes for the nodes we
used the Gain Ratio. The Gain Ration is calculated by dividing the Information Gain by the Split Information.
This will ensure that attributes with fewer values are preffered over those with many values.
2.  When attributes have continuous values , meaning that they have more then 10 different numerical values,
we replaced the values by intervals. The intervals would be chosen in a way that each interval will be calssified in the same way.
If that results in more then 10 intervals, the intervals will be replaced by equally sized intervals that are indipendent of 
the classifications and cover the entire range of values.
3.  as a parameter one can specify a maximum depth for the tree to avoid overfitting and/or keep the tree tidy when using
large data sets with many dimensions

The resulting Tree can be printed (Node class), used to classify data(Test_data class), 
and tested on accuracy with a testing set (Test_data class).

To split the data into testing and training chunks we use the prepare_data function.
The function implements K-folds crossvalidation. So it splits the data into k chunks and then returns
10 different training and testing splits, each time the test set is a different one of the k chunks, the rest is the training set.

Lastly we also implemented a random forest with the Random_forest class. After training it can be used to classify data and can be tested on accuracy.
A rondom forest holds a collection of trees trained on different subsets of the training set. Datapoints are then classified by each tree and the most common classification will be the classification returned by the random forest. Our random forest implemented only the "bag of trees" approach, such that each tree is trained on a different data subset. We did not additionally implement the "bag of features" approach.

The main method builds 10 different decision trees for 4 different datasets (using 10-fold cross validation). 
From the 10 trees the tree with the highest accuracy is then printed.
Aldo a random forest is implemented for each training set. The accuracy of the best decision tree is compared to the accuracy of the random forest.
